import os
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from typing import Dict, List, Tuple, Optional
import random
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2


def collate_fn_filter_none(batch):
    filtered_batch = [item for item in batch if item is not None]

    if len(filtered_batch) == 0:
        return {
            'image': torch.empty(0, 3, 224, 224),
            'edge_map': torch.empty(0, 1, 224, 224),
            'multi_scale_targets': {},
            'image_id': [],
            'original_size': torch.empty(0, 2),
            'base_image_id': [],
            'augmentation_id': []
        }

    from torch.utils.data.dataloader import default_collate
    return default_collate(filtered_batch)


class BSDS500Dataset(Dataset):

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 img_size: Tuple[int, int] = (224, 224),
                 augmentation: bool = True,
                 edge_threshold: float = 0.1,
                 augmentation_factor: int = 10,
                 advanced_augmentation: bool = True,
                 min_edge_ratio: float = 0.001,
                 albumentations_prob: float = 0.9):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augmentation = augmentation and (split == 'train')
        self.edge_threshold = edge_threshold
        self.augmentation_factor = augmentation_factor if (split == 'train' and augmentation) else 1
        self.advanced_augmentation = advanced_augmentation and (split == 'train')
        self.min_edge_ratio = min_edge_ratio
        self.albumentations_prob = albumentations_prob if (split == 'train') else 0.0

        self.images_dir = os.path.join(data_root, 'images', split)
        self.gt_dir = os.path.join(data_root, 'groundTruth', split)

        self.base_image_files = self._get_image_files()

        self.image_files = []
        for img_file in self.base_image_files:
            if self.split == 'train' and self.augmentation:
                for aug_id in range(self.augmentation_factor):
                    self.image_files.append((img_file, aug_id))
            else:
                self.image_files.append((img_file, 0))

        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._init_albumentations_transforms()

        self.bad_data_count = 0
        self.total_access_count = 0

        print(f"BSDS500 {split} dataset initialized (Albumentations Enhanced):")
        print(f"  Base images: {len(self.base_image_files)}")
        print(f"  Total samples (with augmentation): {len(self.image_files)}")
        print(f"  Augmentation factor: {self.augmentation_factor}")
        print(f"  Advanced augmentation (Albumentations): {self.advanced_augmentation}")
        print(f"  Albumentations probability: {self.albumentations_prob}")
        print(f"  Min edge ratio threshold: {self.min_edge_ratio}")

    def _init_albumentations_transforms(self):
        if not self.augmentation or not self.advanced_augmentation:
            self.albumentations_transform = None
            return

        geometric_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_REFLECT),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=10,
                p=0.4,
                border_mode=cv2.BORDER_REFLECT
            ),
            A.ElasticTransform(
                alpha=120,
                sigma=120,
                alpha_affine=80,
                p=0.3,
                border_mode=cv2.BORDER_REFLECT
            ),
            A.GridDistortion(p=0.2),
            A.Perspective(p=0.2),
        ]

        photometric_transforms = [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=0.3
            ),
            A.MotionBlur(
                blur_limit=5,
                p=0.2
            ),
            A.GaussNoise(
                var_limit=(10.0, 80.0),
                p=0.3
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.RandomFog(p=0.1),
            A.RandomShadow(p=0.1),
            A.RandomSnow(p=0.05),
            A.RandomRain(p=0.05),
            A.ISONoise(p=0.2),
            A.Sharpen(p=0.2),
        ]

        all_transforms = geometric_transforms + photometric_transforms

        self.albumentations_transform = A.Compose(
            all_transforms,
            additional_targets={'mask': 'mask'},
            p=self.albumentations_prob
        )

        print(f"  Albumentations augmentation pipeline initialized: {len(all_transforms)} transforms, overall probability={self.albumentations_prob}")

    def _get_image_files(self) -> List[str]:
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        image_files = []
        for filename in os.listdir(self.images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(filename)[0]
                gt_file = os.path.join(self.gt_dir, f"{img_id}.mat")
                if os.path.exists(gt_file):
                    image_files.append(filename)

        return sorted(image_files)

    def _load_ground_truth(self, gt_path: str) -> Optional[np.ndarray]:
        try:
            mat_data = sio.loadmat(gt_path)

            if 'groundTruth' in mat_data:
                gt_data = mat_data['groundTruth']

                if gt_data.size > 0:
                    boundaries = gt_data[0, 0]['Boundaries'][0, 0]

                    if gt_data.shape[1] > 1:
                        combined_boundaries = np.zeros_like(boundaries, dtype=np.float32)
                        for i in range(gt_data.shape[1]):
                            try:
                                boundary = gt_data[0, i]['Boundaries'][0, 0]
                                combined_boundaries = np.maximum(combined_boundaries, boundary.astype(np.float32))
                            except:
                                continue

                        if combined_boundaries.max() > 0:
                            return combined_boundaries
                        else:
                            print(f"Warning: Combined boundaries are empty for {gt_path}")
                            return None
                    else:
                        if boundaries.max() > 0:
                            return boundaries.astype(np.float32)
                        else:
                            print(f"Warning: Single boundary is empty for {gt_path}")
                            return None

            for key in ['bmap', 'seg', 'boundaries']:
                if key in mat_data:
                    data = mat_data[key].astype(np.float32)
                    if data.max() > 0:
                        return data
                    else:
                        print(f"Warning: {key} field is empty for {gt_path}")

            print(f"Error: No valid ground truth data found in {gt_path}")
            return None

        except Exception as e:
            print(f"Error loading ground truth from {gt_path}: {e}")
            return None

    def _is_valid_edge_map(self, edge_map: np.ndarray) -> bool:
        if edge_map is None:
            return False

        if edge_map.max() == 0:
            return False

        edge_pixels = np.sum(edge_map > self.edge_threshold)
        total_pixels = edge_map.size
        edge_ratio = edge_pixels / total_pixels

        if edge_ratio < self.min_edge_ratio:
            print(f"Warning: Edge ratio {edge_ratio:.6f} below threshold {self.min_edge_ratio}")
            return False

        if not isinstance(edge_map, np.ndarray):
            return False

        if len(edge_map.shape) != 2:
            return False

        if edge_map.min() < 0 or edge_map.max() > 1:
            if edge_map.max() > edge_map.min():
                edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())
            else:
                return False

        return True

    def _process_edge_map(self, edge_map: np.ndarray) -> Optional[np.ndarray]:
        try:
            edge_map = cv2.resize(edge_map, self.img_size, interpolation=cv2.INTER_LINEAR)
            if edge_map.max() > 1.0:
                edge_map = edge_map / edge_map.max()
            edge_map = np.clip(edge_map, 0.0, 1.0)
            if not self._is_valid_edge_map(edge_map):
                return None
            return edge_map.astype(np.float32)
        except Exception as e:
            print(f"Error processing edge map: {e}")
            return None

    def _create_multi_scale_targets(self, edge_map: np.ndarray) -> Dict[str, torch.Tensor]:
        targets = {}
        targets['full'] = torch.from_numpy(edge_map).unsqueeze(0).float()
        scales = [0.5, 0.25, 0.125]
        for i, scale in enumerate(scales):
            h, w = int(self.img_size[0] * scale), int(self.img_size[1] * scale)
            scaled_edge = cv2.resize(edge_map, (w, h), interpolation=cv2.INTER_LINEAR)
            targets[f'scale_{i}'] = torch.from_numpy(scaled_edge).unsqueeze(0).float()
        return targets

    def _apply_albumentations_augmentation(self, image_np: np.ndarray, edge_map: np.ndarray, aug_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augmentation or not self.advanced_augmentation or self.albumentations_transform is None:
            return image_np, edge_map

        try:
            random.seed(hash((id(image_np), aug_id, "albumentations")) % (2 ** 32))
            np.random.seed(hash((id(image_np), aug_id, "albumentations")) % (2 ** 32))
            edge_map_aug = (edge_map * 255).astype(np.uint8)
            augmented = self.albumentations_transform(image=image_np, mask=edge_map_aug)
            image_aug = augmented['image']
            edge_map_aug = augmented['mask'].astype(np.float32) / 255.0
            if not self._is_valid_edge_map(edge_map_aug):
                print(f"Warning: Invalid edge map after Albumentations augmentation, using original")
                return image_np, edge_map
            return image_aug, edge_map_aug
        except Exception as e:
            print(f"Error in Albumentations augmentation: {e}, using original data")
            return image_np, edge_map

    def _apply_basic_augmentation(self, image: Image.Image, edge_map: np.ndarray, aug_id: int) -> Tuple[Image.Image, np.ndarray]:
        if not self.augmentation:
            return image, edge_map

        img_np = np.array(image)
        random.seed(hash((id(image), aug_id, "basic")) % (2 ** 32))
        np.random.seed(hash((id(image), aug_id, "basic")) % (2 ** 32))

        try:
            if random.random() > 0.5:
                img_np = np.fliplr(img_np)
                edge_map = np.fliplr(edge_map)
            if random.random() > 0.7:
                angle = random.uniform(-5, 5)
                h, w = img_np.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_np = cv2.warpAffine(img_np, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                edge_map = cv2.warpAffine(edge_map, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return Image.fromarray(img_np), edge_map
        except Exception as e:
            print(f"Error in basic augmentation: {e}")
            return image, edge_map

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        self.total_access_count += 1

        try:
            img_filename, aug_id = self.image_files[idx]
            img_id = os.path.splitext(img_filename)[0]
            img_path = os.path.join(self.images_dir, img_filename)
            gt_path = os.path.join(self.gt_dir, f"{img_id}.mat")

            try:
                image = Image.open(img_path).convert('RGB')
                if image.size[0] == 0 or image.size[1] == 0:
                    print(f"Error: Invalid image size for {img_path}")
                    self.bad_data_count += 1
                    return None
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                self.bad_data_count += 1
                return None

            edge_map = self._load_ground_truth(gt_path)
            if edge_map is None:
                print(f"Error: Failed to load valid GT for {img_id}")
                self.bad_data_count += 1
                return None

            if not self._is_valid_edge_map(edge_map):
                print(f"Error: Invalid original edge map for {img_id}")
                self.bad_data_count += 1
                return None

            image = image.resize(self.img_size, Image.LANCZOS)
            edge_map = cv2.resize(edge_map, self.img_size, interpolation=cv2.INTER_LINEAR)

            if self.split == 'train' and self.augmentation:
                image_np = np.array(image)
                if self.advanced_augmentation and self.albumentations_transform is not None:
                    image_np, edge_map = self._apply_albumentations_augmentation(image_np, edge_map, aug_id)
                    image = Image.fromarray(image_np)
                else:
                    image, edge_map = self._apply_basic_augmentation(image, edge_map, aug_id)

            if not self._is_valid_edge_map(edge_map):
                print(f"Error: Invalid edge map after processing for {img_id}")
                self.bad_data_count += 1
                return None

            edge_map = self._process_edge_map(edge_map)
            if edge_map is None:
                print(f"Error: Failed to process edge map for {img_id}")
                self.bad_data_count += 1
                return None

            if not self._is_valid_edge_map(edge_map):
                print(f"Error: Final edge map validation failed for {img_id}")
                self.bad_data_count += 1
                return None

            image_tensor = self.normalize_transform(image)
            targets = self._create_multi_scale_targets(edge_map)

            if idx % 100 == 0:
                print(f"Sample {idx}: Split={self.split}, Aug={self.augmentation}, AdvAug={self.advanced_augmentation}, "
                      f"EdgeMax={edge_map.max():.2f}, EdgeRatio={np.mean(edge_map > 0.1):.4f}")

            return {
                'image': image_tensor,
                'edge_map': targets['full'],
                'multi_scale_targets': targets,
                'image_id': f"{img_id}_aug{aug_id}",
                'original_size': torch.tensor([image.size[1], image.size[0]]),
                'base_image_id': img_id,
                'augmentation_id': aug_id
            }

        except Exception as e:
            print(f"Unexpected error in __getitem__ for idx {idx}: {e}")
            self.bad_data_count += 1
            return None

    def get_statistics(self) -> Dict[str, float]:
        return {
            'total_access': self.total_access_count,
            'bad_data_count': self.bad_data_count,
            'bad_data_ratio': self.bad_data_count / max(self.total_access_count, 1),
            'valid_data_ratio': 1 - (self.bad_data_count / max(self.total_access_count, 1))
        }


class BSDS500DataModule:

    def __init__(self,
                 data_root: str,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 img_size: Tuple[int, int] = (224, 224),
                 pin_memory: bool = True,
                 augmentation_factor: int = 5,
                 advanced_augmentation: bool = True,
                 min_edge_ratio: float = 0.001,
                 albumentations_prob: float = 0.8):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.pin_memory = pin_memory
        self.augmentation_factor = augmentation_factor
        self.advanced_augmentation = advanced_augmentation
        self.min_edge_ratio = min_edge_ratio
        self.albumentations_prob = albumentations_prob

        self._validate_dataset()

    def _validate_dataset(self):
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'groundTruth/train', 'groundTruth/val', 'groundTruth/test'
        ]
        for dir_path in required_dirs:
            full_path = os.path.join(self.data_root, dir_path)
            if not os.path.exists(full_path):
                print(f"Warning: Directory not found: {full_path}")

    def get_dataloader(self, split: str = 'train', shuffle: bool = None) -> DataLoader:
        if shuffle is None:
            shuffle = (split == 'train')

        use_augmentation = (split == 'train')
        current_aug_factor = self.augmentation_factor if use_augmentation else 1

        dataset = BSDS500Dataset(
            data_root=self.data_root,
            split=split,
            img_size=self.img_size,
            augmentation=use_augmentation,
            edge_threshold=0.1,
            augmentation_factor=current_aug_factor,
            advanced_augmentation=self.advanced_augmentation,
            min_edge_ratio=self.min_edge_ratio,
            albumentations_prob=self.albumentations_prob
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(split == 'train'),
            collate_fn=collate_fn_filter_none
        )

    def get_eval_dataloader(self, split: str = 'test', shuffle: bool = False) -> DataLoader:
        dataset = BSDS500Dataset(
            data_root=self.data_root,
            split=split,
            img_size=self.img_size,
            augmentation=False,
            edge_threshold=0.1,
            augmentation_factor=1,
            advanced_augmentation=False,
            min_edge_ratio=self.min_edge_ratio,
            albumentations_prob=0.0
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn_filter_none
        )

    def get_all_dataloaders(self) -> Dict[str, DataLoader]:
        return {
            'train': self.get_dataloader('train', shuffle=True),
            'val': self.get_dataloader('val', shuffle=False),
            'test': self.get_dataloader('test', shuffle=False)
        }

    def get_dataset_info(self) -> Dict[str, int]:
        info = {}
        for split in ['train', 'val', 'test']:
            try:
                use_augmentation = (split == 'train')
                current_aug_factor = self.augmentation_factor if use_augmentation else 1

                dataset = BSDS500Dataset(
                    data_root=self.data_root,
                    split=split,
                    img_size=self.img_size,
                    augmentation=use_augmentation,
                    edge_threshold=0.1,
                    augmentation_factor=current_aug_factor,
                    advanced_augmentation=self.advanced_augmentation,
                    min_edge_ratio=self.min_edge_ratio,
                    albumentations_prob=self.albumentations_prob
                )
                info[split] = len(dataset)
            except:
                info[split] = 0
        return info
