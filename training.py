import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Literal

from model import DualBranchEdgeModel
from dataset import BSDS500DataModule
from edge_losses import UnifiedEdgeDetectionLoss, LossScheduler, create_loss_with_scheduler


def setup_logger(log_dir, name, file_name='train_log.txt'):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(log_dir, file_name))
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class EdgeDetectionTrainer:

    def __init__(self,
                 model,
                 data_module,
                 optimizer,
                 scheduler=None,
                 loss_fn=None,
                 loss_scheduler=None,
                 device=None,
                 config=None,
                 checkpoint_dir='checkpoints',
                 log_dir='logs',
                 dataset_name='unknown'):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset_name = dataset_name

        self.loss_fn = loss_fn if loss_fn is not None else UnifiedEdgeDetectionLoss()
        self.loss_scheduler = loss_scheduler

        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        self.config = config or {}
        self.epochs = self.config.get('epochs', 100)
        self.grad_clip_value = self.config.get('grad_clip_value', 1.0)
        self.accumulation_steps = self.config.get('accumulation_steps', 1)
        self.use_amp = self.config.get('use_amp', True) and self.device.type == 'cuda'
        self.checkpoint_interval = self.config.get('checkpoint_interval', 10)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 20)
        self.min_delta = self.config.get('min_delta', 1e-4)

        self.scaler = torch.amp.GradScaler('cuda',
                                           enabled=self.use_amp) if self.device.type == 'cuda' else torch.amp.GradScaler(
            'cpu', enabled=False)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.log_dir = log_dir
        self.logger = setup_logger(self.log_dir, 'edge_detection_trainer',
                                   file_name=f'train_log_{dataset_name}.txt')

        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(f"Model: {type(model).__name__}")
        self.logger.info(f"Optimizer: {type(optimizer).__name__}")
        self.logger.info(f"LR Scheduler: {type(scheduler).__name__ if scheduler else 'None'}")
        self.logger.info(f"Loss function: {type(self.loss_fn).__name__}")
        self.logger.info(f"Loss Scheduler: {type(self.loss_scheduler).__name__ if self.loss_scheduler else 'None'}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training config: {self.config}")

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def train_epoch(self, epoch):
        self.model.train()
        train_loader = self.data_module.get_dataloader('train')

        epoch_loss = 0
        loss_components_sum = {}
        batch_count = 0

        self.optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [{self.dataset_name}]", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            if batch['image'].numel() == 0:
                continue

            images = batch['image'].to(self.device)
            edge_maps = batch['edge_map'].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_components = self.loss_fn(outputs, edge_maps)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.grad_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps
            batch_count += 1

            for k, v in loss_components.items():
                loss_components_sum[k] = loss_components_sum.get(k, 0) + v.item()

            progress_bar.set_postfix(loss=epoch_loss / batch_count if batch_count > 0 else 0)

        if batch_count == 0:
            self.logger.warning(f"Epoch {epoch} train loader was empty.")
            return 0.0, {}

        avg_loss = epoch_loss / batch_count
        avg_components = {k: v / batch_count for k, v in loss_components_sum.items()}

        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Epoch {epoch} Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        if epoch % 10 == 0:
            for k, v in avg_components.items():
                if 'loss' in k:
                    self.logger.debug(f"  Train {k}: {v:.4f}")

        return avg_loss, avg_components

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loader = self.data_module.get_dataloader('val')

        if len(val_loader) == 0:
            self.logger.warning(f"Epoch {epoch} validation loader is empty. Skipping validation.")
            return float('inf'), {}

        val_loss = 0
        val_components_sum = {}
        batch_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation [{self.dataset_name}]", leave=False):
                if batch['image'].numel() == 0:
                    continue

                images = batch['image'].to(self.device)
                edge_maps = batch['edge_map'].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss, loss_components = self.loss_fn(outputs, edge_maps)

                val_loss += loss.item()
                batch_count += 1

                for k, v in loss_components.items():
                    val_components_sum[k] = val_components_sum.get(k, 0) + v.item()

        if batch_count == 0:
            return float('inf'), {}

        avg_val_loss = val_loss / batch_count
        avg_val_components = {k: v / batch_count for k, v in val_components_sum.items()}

        self.logger.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

        if epoch % 10 == 0:
            for k, v in avg_val_components.items():
                if 'loss' in k:
                    self.logger.debug(f"  Val {k}: {v:.4f}")

        return avg_val_loss, avg_val_components

    def save_checkpoint(self, epoch, val_loss, is_best=False, final=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'dataset_name': self.dataset_name,
            'config': self.config,
            'training_history': self.training_history
        }

        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, f"model_best_{self.dataset_name}.pth")
            torch.save(state, best_filename)
            self.logger.info(f"Best model saved: {best_filename} (Val Loss: {val_loss:.4f})")

        if epoch % self.checkpoint_interval == 0:
            filename = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}_{self.dataset_name}.pth")
            torch.save(state, filename)
            self.logger.info(f"Checkpoint saved: {filename}")

        if final:
            final_filename = os.path.join(self.checkpoint_dir, f"model_final_{self.dataset_name}.pth")
            torch.save(state, final_filename)
            self.logger.info(f"Final model saved: {final_filename}")

        latest_filename = os.path.join(self.checkpoint_dir, f"model_latest_{self.dataset_name}.pth")
        torch.save(state, latest_filename)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint not found at {path}, starting from scratch.")
            return 1

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        start_epoch = checkpoint.get('epoch', 0) + 1
        self.logger.info(f"Resumed from checkpoint {path} at epoch {start_epoch}.")
        return start_epoch

    def train(self, resume_from=None):
        self.logger.info(f"=== Starting training on {self.dataset_name} ===")
        self.model.to(self.device)

        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        best_val_loss = float('inf')

        epoch = 0
        val_loss = float('inf')

        try:
            for epoch in range(start_epoch, self.epochs + 1):
                if self.loss_scheduler is not None:
                    self.loss_scheduler.step(epoch)
                    if epoch % 5 == 0 or epoch == 1:
                        current_weights = self.loss_scheduler.get_current_weights()
                        weights_str = ", ".join(
                            [f"{k.replace('_weight', '')}:{v:.3f}" for k, v in current_weights.items()])
                        self.logger.info(f"Epoch {epoch} Loss Weights: [{weights_str}]")

                train_loss, train_components = self.train_epoch(epoch)
                self.training_history['train_loss'].append(train_loss)

                val_loss, val_components = self.validate_epoch(epoch)
                self.training_history['val_loss'].append(val_loss)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)

                if self.scheduler is not None:
                    self.scheduler.step()

                is_best = False
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    self.patience_counter = 0
                    is_best = True
                else:
                    self.patience_counter += 1

                self.save_checkpoint(epoch, val_loss, is_best)

                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    self.logger.info(f"Progress: Epoch {epoch}/{self.epochs}, "
                                     f"Best Val Loss: {best_val_loss:.4f}, "
                                     f"Patience: {self.patience_counter}/{self.early_stopping_patience}")

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            if epoch == 0:
                self.logger.error("Failed before completing any epoch")
                return float('inf'), None
            raise e

        finally:
            if epoch > 0:
                self.save_checkpoint(epoch, val_loss, final=True)

        self.logger.info(f"=== Training completed on {self.dataset_name}. Best Val Loss: {best_val_loss:.4f} ===")
        return best_val_loss, self.checkpoint_dir


def create_data_module(dataset: str,
                       data_root: str,
                       annotation_type: str = 'edge',
                       batch_size: int = 8,
                       img_size: int = 224,
                       num_workers: int = 4,
                       augmentation_factor: int = 10):

    if dataset.lower() == 'bsds500':
        return BSDS500DataModule(
            data_root=data_root,
            batch_size=batch_size,
            img_size=(img_size, img_size),
            num_workers=num_workers,
            augmentation_factor=augmentation_factor,
            advanced_augmentation=True,
            albumentations_prob=0.9
        ), 'BSDS500'


    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def train_edge_detection_model(
        dataset: str = 'bsds500',
        data_root: str = None,
        annotation_type: str = 'edge',
        output_dir: str = './output',
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        img_size: int = 224,
        resume_from: Optional[str] = None,
        loss_schedule_type: str = 'dynamic'
):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"{dataset}_{annotation_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    data_module, dataset_name = create_data_module(
        dataset=dataset,
        data_root=data_root,
        annotation_type=annotation_type,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=4,
        augmentation_factor=10
    )

    model = DualBranchEdgeModel(img_size=img_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    loss_fn, loss_scheduler = create_loss_with_scheduler(
        total_epochs=epochs,
        schedule_type=loss_schedule_type,
        main_weight=1.0,
        side_weight=0.4,
        bce_weight=0.3,
        focal_weight=0.5,
        dice_weight=0.8,
        tversky_weight=0.4,
        continuity_weight=1.0,
        direction_weight=1.0,
        gradient_weight=1.5,
        auto_balance=True,
        edge_enhancement=True
    )

    config = {
        'epochs': epochs,
        'grad_clip_value': 1.0,
        'accumulation_steps': 4,
        'use_amp': True,
        'checkpoint_interval': 10,
        'early_stopping_patience': 20,
        'min_delta': 1e-4
    }

    trainer = EdgeDetectionTrainer(
        model=model,
        data_module=data_module,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        loss_scheduler=loss_scheduler,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        config=config,
        checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
        log_dir=os.path.join(output_dir, 'logs'),
        dataset_name=dataset_name
    )

    best_loss, checkpoint_dir = trainer.train(resume_from=resume_from)

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")

    return checkpoint_dir
