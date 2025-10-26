import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        true_positive = (pred_flat * target_flat).sum(dim=1)
        false_positive = (pred_flat * (1 - target_flat)).sum(dim=1)
        false_negative = ((1 - pred_flat) * target_flat).sum(dim=1)
        tversky_index = (true_positive + self.smooth) / (
                true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        tversky_loss = 1.0 - tversky_index

        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


class BalancedBCELoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean', auto_weight=True):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.reduction = reduction
        self.auto_weight = auto_weight

    def forward(self, pred, target):
        if self.auto_weight and self.pos_weight_val is None:
            pos_count = target.sum()
            neg_count = target.numel() - pos_count

            if pos_count > 0:
                pos_weight = neg_count / pos_count
                pos_weight = torch.clamp(pos_weight, min=0.5, max=10.0)
            else:
                pos_weight = torch.tensor(1.0, device=target.device)
        elif self.pos_weight_val is not None:
            pos_weight = torch.tensor(self.pos_weight_val, device=target.device)
        else:
            pos_weight = None

        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight, reduction=self.reduction)


class GradientLoss(nn.Module):
    def __init__(self,
                 weight=1.0,
                 gradient_type='sobel',
                 magnitude_weight=0.6,
                 direction_weight=0.4,
                 edge_threshold=0.1,
                 sharpness_penalty=1.2):
        super().__init__()
        self.weight = weight
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        self.edge_threshold = edge_threshold
        self.sharpness_penalty = sharpness_penalty

        if gradient_type == 'sobel':
            self.grad_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                       dtype=torch.float32).view(1, 1, 3, 3)
            self.grad_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                       dtype=torch.float32).view(1, 1, 3, 3)
        elif gradient_type == 'scharr':
            self.grad_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                       dtype=torch.float32).view(1, 1, 3, 3)
            self.grad_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                       dtype=torch.float32).view(1, 1, 3, 3)
        elif gradient_type == 'prewitt':
            self.grad_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                                       dtype=torch.float32).view(1, 1, 3, 3)
            self.grad_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                                       dtype=torch.float32).view(1, 1, 3, 3)
        else:
            raise ValueError(f"Unsupported gradient type: {gradient_type}")

    def _compute_gradients(self, image):
        grad_x = F.conv2d(image, self.grad_x, padding=1)
        grad_y = F.conv2d(image, self.grad_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        direction = torch.atan2(grad_y, grad_x + 1e-6)
        return grad_x, grad_y, magnitude, direction

    def _compute_sharpness_measure(self, image):
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                        dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
        laplacian = F.conv2d(image, laplacian_kernel, padding=1)
        return torch.var(laplacian, dim=[2, 3], keepdim=True)

    def forward(self, pred, target):
        device = pred.device
        self.grad_x = self.grad_x.to(device)
        self.grad_y = self.grad_y.to(device)

        pred_prob = torch.sigmoid(pred)

        pred_grad_x, pred_grad_y, pred_magnitude, pred_direction = self._compute_gradients(pred_prob)
        target_grad_x, target_grad_y, target_magnitude, target_direction = self._compute_gradients(target)

        magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude, reduction='none')

        pred_grad_norm_x = pred_grad_x / (pred_magnitude + 1e-6)
        pred_grad_norm_y = pred_grad_y / (pred_magnitude + 1e-6)
        target_grad_norm_x = target_grad_x / (target_magnitude + 1e-6)
        target_grad_norm_y = target_grad_y / (target_magnitude + 1e-6)

        direction_consistency = (pred_grad_norm_x * target_grad_norm_x +
                                 pred_grad_norm_y * target_grad_norm_y)
        direction_loss = 1.0 - direction_consistency
        direction_loss = torch.clamp(direction_loss, min=0.0, max=2.0)

        edge_mask = (target_magnitude > self.edge_threshold).float()

        pred_sharpness = self._compute_sharpness_measure(pred_prob)
        target_sharpness = self._compute_sharpness_measure(target)
        sharpness_loss = F.mse_loss(pred_sharpness, target_sharpness, reduction='mean')

        weighted_magnitude_loss = magnitude_loss * edge_mask
        weighted_direction_loss = direction_loss * edge_mask

        valid_pixels = torch.sum(edge_mask, dim=[2, 3], keepdim=True) + 1e-6

        magnitude_loss_normalized = torch.sum(weighted_magnitude_loss, dim=[2, 3], keepdim=True) / valid_pixels
        direction_loss_normalized = torch.sum(weighted_direction_loss, dim=[2, 3], keepdim=True) / valid_pixels

        gradient_loss = (self.magnitude_weight * magnitude_loss_normalized.mean() +
                         self.direction_weight * direction_loss_normalized.mean() +
                         self.sharpness_penalty * sharpness_loss)

        return gradient_loss * self.weight


class EdgeContinuityLoss(nn.Module):
    def __init__(self, weight=1.0, window_size=3, smoothness_weight=0.5):
        super().__init__()
        self.weight = weight
        self.window_size = window_size
        self.smoothness_weight = smoothness_weight

        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)

        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                      dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.laplacian = self.laplacian.to(device)

        pred_prob = torch.sigmoid(pred)

        batch_size = pred.shape[0]

        pred_grad_x = F.conv2d(pred_prob, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_prob, self.sobel_y, padding=1)

        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        pred_grad_magnitude = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        target_grad_magnitude = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)

        pred_laplacian = F.conv2d(pred_prob, self.laplacian, padding=1)
        target_laplacian = F.conv2d(target, self.laplacian, padding=1)

        pred_edge_mask = (pred_grad_magnitude > 0.1).float()
        target_edge_mask = (target_grad_magnitude > 0.1).float()

        continuity_loss = torch.abs(pred_laplacian - target_laplacian) * target_edge_mask

        window = self.window_size

        pred_max_pool = F.max_pool2d(pred_grad_magnitude, window, stride=1, padding=window // 2)
        pred_avg_pool = F.avg_pool2d(pred_grad_magnitude, window, stride=1, padding=window // 2)

        smoothness_loss = torch.abs(pred_max_pool - pred_avg_pool) * pred_edge_mask

        combined_loss = continuity_loss.mean() + self.smoothness_weight * smoothness_loss.mean()

        return combined_loss * self.weight


class EdgeDirectionConsistencyLoss(nn.Module):
    def __init__(self, weight=1.0, angle_threshold=0.7):
        super().__init__()
        self.weight = weight
        self.angle_threshold = angle_threshold

        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        pred_prob = torch.sigmoid(pred)

        pred_grad_x = F.conv2d(pred_prob, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_prob, self.sobel_y, padding=1)

        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        pred_grad_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)
        target_grad_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)

        pred_grad_x_norm = pred_grad_x / (pred_grad_mag + 1e-6)
        pred_grad_y_norm = pred_grad_y / (pred_grad_mag + 1e-6)

        target_grad_x_norm = target_grad_x / (target_grad_mag + 1e-6)
        target_grad_y_norm = target_grad_y / (target_grad_mag + 1e-6)

        direction_cosine = pred_grad_x_norm * target_grad_x_norm + pred_grad_y_norm * target_grad_y_norm

        edge_mask = (target_grad_mag > 0.1).float()

        direction_inconsistency = 1.0 - direction_cosine

        direction_loss = direction_inconsistency * edge_mask

        weighted_direction_loss = direction_loss * target_grad_mag

        valid_pixels = torch.sum(edge_mask) + 1e-6
        loss = torch.sum(weighted_direction_loss) / valid_pixels

        return loss * self.weight


class UnifiedEdgeDetectionLoss(nn.Module):
    def __init__(self,
                 main_weight=1.0,
                 side_weight=0.5,

                 bce_weight=0.3,
                 focal_weight=0.5,
                 dice_weight=0.8,
                 tversky_weight=0.4,

                 continuity_weight=1.0,
                 direction_weight=1.0,
                 gradient_weight=1.5,

                 focal_alpha=0.25,
                 focal_gamma=2.0,

                 tversky_alpha=0.7,
                 tversky_beta=0.3,

                 continuity_window_size=3,
                 continuity_smoothness=0.5,

                 direction_angle_threshold=0.7,

                 gradient_type='sobel',
                 gradient_magnitude_weight=0.6,
                 gradient_direction_weight=0.4,
                 gradient_edge_threshold=0.1,
                 gradient_sharpness_penalty=1.2,

                 auto_balance=True,
                 edge_enhancement=True):
        super().__init__()

        self.main_weight = main_weight
        self.side_weight = side_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.continuity_weight = continuity_weight
        self.direction_weight = direction_weight
        self.gradient_weight = gradient_weight

        self.auto_balance = auto_balance
        self.edge_enhancement = edge_enhancement

        total_weight = (bce_weight + focal_weight + dice_weight + tversky_weight +
                        continuity_weight + direction_weight + gradient_weight)

        if total_weight > 0:
            self.bce_weight = bce_weight / total_weight
            self.focal_weight = focal_weight / total_weight
            self.dice_weight = dice_weight / total_weight
            self.tversky_weight = tversky_weight / total_weight
            self.continuity_weight = continuity_weight / total_weight
            self.direction_weight = direction_weight / total_weight
            self.gradient_weight = gradient_weight / total_weight

        self.bce_loss = BalancedBCELoss(auto_weight=auto_balance, reduction='mean')
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

        self.continuity_loss = EdgeContinuityLoss(
            weight=1.0,
            window_size=continuity_window_size,
            smoothness_weight=continuity_smoothness
        )

        self.direction_loss = EdgeDirectionConsistencyLoss(
            weight=1.0,
            angle_threshold=direction_angle_threshold
        )

        self.gradient_loss = GradientLoss(
            weight=1.0,
            gradient_type=gradient_type,
            magnitude_weight=gradient_magnitude_weight,
            direction_weight=gradient_direction_weight,
            edge_threshold=gradient_edge_threshold,
            sharpness_penalty=gradient_sharpness_penalty
        )

        self.edge_weight_factor = 10.0

    def _compute_single_output_loss(self, pred, target):
        losses = {}
        device = pred.device

        target = target.to(device).float()

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("Warning: NaN or Inf detected in predictions")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(target).any() or torch.isinf(target).any():
            print("Warning: NaN or Inf detected in targets")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        if self.bce_weight > 0:
            try:
                bce = self.bce_loss(pred, target)
                if not torch.isnan(bce) and not torch.isinf(bce):
                    losses['bce'] = bce * self.bce_weight
                else:
                    losses['bce'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"BCE loss error: {e}")
                losses['bce'] = torch.tensor(0.0, device=device)

        if self.focal_weight > 0:
            try:
                focal = self.focal_loss(pred, target)
                if not torch.isnan(focal) and not torch.isinf(focal):
                    losses['focal'] = focal * self.focal_weight
                else:
                    losses['focal'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Focal loss error: {e}")
                losses['focal'] = torch.tensor(0.0, device=device)

        if self.dice_weight > 0:
            try:
                dice = self.dice_loss(pred, target)
                if not torch.isnan(dice) and not torch.isinf(dice):
                    losses['dice'] = dice * self.dice_weight
                else:
                    losses['dice'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Dice loss error: {e}")
                losses['dice'] = torch.tensor(0.0, device=device)

        if self.tversky_weight > 0:
            try:
                tversky = self.tversky_loss(pred, target)
                if not torch.isnan(tversky) and not torch.isinf(tversky):
                    losses['tversky'] = tversky * self.tversky_weight
                else:
                    losses['tversky'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Tversky loss error: {e}")
                losses['tversky'] = torch.tensor(0.0, device=device)

        if self.continuity_weight > 0:
            try:
                continuity = self.continuity_loss(pred, target)
                if not torch.isnan(continuity) and not torch.isinf(continuity):
                    losses['continuity'] = continuity * self.continuity_weight
                else:
                    losses['continuity'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Continuity loss error: {e}")
                losses['continuity'] = torch.tensor(0.0, device=device)

        if self.direction_weight > 0:
            try:
                direction = self.direction_loss(pred, target)
                if not torch.isnan(direction) and not torch.isinf(direction):
                    losses['direction'] = direction * self.direction_weight
                else:
                    losses['direction'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Direction loss error: {e}")
                losses['direction'] = torch.tensor(0.0, device=device)

        if self.gradient_weight > 0:
            try:
                gradient = self.gradient_loss(pred, target)
                if not torch.isnan(gradient) and not torch.isinf(gradient):
                    losses['gradient'] = gradient * self.gradient_weight
                else:
                    losses['gradient'] = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Gradient loss error: {e}")
                losses['gradient'] = torch.tensor(0.0, device=device)

        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

        if self.edge_enhancement:
            try:
                edge_pixels = target.sum()
                if edge_pixels > 0:
                    edge_mask = (target > 0.5).float()
                    edge_loss = (F.binary_cross_entropy_with_logits(
                        pred, target, reduction='none') * edge_mask).sum() / (edge_pixels + 1e-6)

                    safe_edge_weight = min(self.edge_weight_factor, 5.0)
                    total_loss = total_loss + edge_loss * safe_edge_weight * 0.1
            except Exception as e:
                print(f"Edge enhancement error: {e}")

        return total_loss, losses

    def forward(self, predictions, targets):
        targets = targets.float()
        main_pred = predictions['final_edges']
        device = main_pred.device

        if main_pred.shape[-2:] != targets.shape[-2:]:
            targets_resized = F.interpolate(targets, size=main_pred.shape[-2:], mode='bilinear', align_corners=False)
        else:
            targets_resized = targets

        if targets_resized.max() == 0:
            print("Warning: Target is all zeros, this may cause training issues")

        main_loss, main_components = self._compute_single_output_loss(main_pred, targets_resized)

        total_loss = self.main_weight * main_loss
        loss_dict = {
            'total_loss': total_loss.clone(),
            'main_loss': main_loss.clone(),
            'main_bce': main_components.get('bce', torch.tensor(0.0, device=device)).clone(),
            'main_focal': main_components.get('focal', torch.tensor(0.0, device=device)).clone(),
            'main_dice': main_components.get('dice', torch.tensor(0.0, device=device)).clone(),
            'main_tversky': main_components.get('tversky', torch.tensor(0.0, device=device)).clone(),
            'main_continuity': main_components.get('continuity', torch.tensor(0.0, device=device)).clone(),
            'main_direction': main_components.get('direction', torch.tensor(0.0, device=device)).clone(),
            'main_gradient': main_components.get('gradient', torch.tensor(0.0, device=device)).clone()
        }

        side_outputs = predictions.get('side_outputs', [])
        if side_outputs and self.side_weight > 0:
            side_losses = []

            side_components_sum = {
                'bce': torch.tensor(0.0, device=device),
                'focal': torch.tensor(0.0, device=device),
                'dice': torch.tensor(0.0, device=device),
                'tversky': torch.tensor(0.0, device=device),
                'continuity': torch.tensor(0.0, device=device),
                'direction': torch.tensor(0.0, device=device),
                'gradient': torch.tensor(0.0, device=device)
            }

            for i, side_pred in enumerate(side_outputs):
                if side_pred.shape[-2:] != targets.shape[-2:]:
                    targets_scaled = F.interpolate(targets, size=side_pred.shape[-2:], mode='bilinear',
                                                   align_corners=False)
                else:
                    targets_scaled = targets

                side_loss, side_components = self._compute_single_output_loss(side_pred, targets_scaled)
                side_losses.append(side_loss)

                for key, val in side_components.items():
                    if key in side_components_sum:
                        side_components_sum[key] += val

                loss_dict[f'side_{i}_loss'] = side_loss.clone()

            if side_losses:
                avg_side_loss = sum(side_losses) / len(side_losses)
                total_loss += self.side_weight * avg_side_loss

                num_outputs = len(side_outputs)
                loss_dict.update({
                    'side_loss_avg': avg_side_loss.clone(),
                    'side_bce_avg': (side_components_sum['bce'] / num_outputs).clone(),
                    'side_focal_avg': (side_components_sum['focal'] / num_outputs).clone(),
                    'side_dice_avg': (side_components_sum['dice'] / num_outputs).clone(),
                    'side_tversky_avg': (side_components_sum['tversky'] / num_outputs).clone(),
                    'side_continuity_avg': (side_components_sum['continuity'] / num_outputs).clone(),
                    'side_direction_avg': (side_components_sum['direction'] / num_outputs).clone(),
                    'side_gradient_avg': (side_components_sum['gradient'] / num_outputs).clone()
                })

        loss_dict['total_loss'] = total_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN or Inf in total loss, setting to zero")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if not total_loss.requires_grad:
            print("Warning: total_loss doesn't require grad, creating a new tensor with grad")
            total_loss = total_loss.clone().detach().requires_grad_(True)

        return total_loss, loss_dict


class LossScheduler:
    def __init__(self, loss_fn, schedule_type='dynamic', total_epochs=100):
        self.loss_fn = loss_fn
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs

        self.target_weights = {
            'bce_weight': loss_fn.bce_weight,
            'focal_weight': loss_fn.focal_weight,
            'dice_weight': loss_fn.dice_weight,
            'tversky_weight': loss_fn.tversky_weight,
            'continuity_weight': loss_fn.continuity_weight,
            'direction_weight': loss_fn.direction_weight,
            'gradient_weight': loss_fn.gradient_weight
        }

        self.initial_weights = {
            'bce_weight': 1.0,
            'focal_weight': 1.2,
            'dice_weight': 0.8,
            'tversky_weight': 0.4,
            'continuity_weight': 0.3,
            'direction_weight': 0.2,
            'gradient_weight': 0.5
        }

        self.mid_weights = {
            'bce_weight': 0.6,
            'focal_weight': 0.8,
            'dice_weight': 1.0,
            'tversky_weight': 0.6,
            'continuity_weight': 0.8,
            'direction_weight': 0.7,
            'gradient_weight': 1.2
        }

        self._normalize_weights(self.initial_weights)
        self._normalize_weights(self.mid_weights)
        self._normalize_weights(self.target_weights)

    def _normalize_weights(self, weights):
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] = weights[key] / total

    def step(self, epoch):
        if self.schedule_type == 'dynamic':
            self._dynamic_schedule(epoch)
        elif self.schedule_type == 'cosine':
            self._cosine_schedule(epoch)
        elif self.schedule_type == 'linear':
            self._linear_schedule(epoch)
        elif self.schedule_type == 'step':
            self._step_schedule(epoch)

        self._apply_current_weights()

    def _dynamic_schedule(self, epoch):
        progress = epoch / self.total_epochs

        if progress < 0.25:
            current_weights = self.initial_weights.copy()

        elif progress < 0.70:
            transition_progress = (progress - 0.25) / 0.45
            current_weights = {}
            for key in self.initial_weights:
                current_weights[key] = (
                        self.initial_weights[key] * (1 - transition_progress) +
                        self.mid_weights[key] * transition_progress
                )

        else:
            transition_progress = (progress - 0.70) / 0.30
            current_weights = {}
            for key in self.mid_weights:
                current_weights[key] = (
                        self.mid_weights[key] * (1 - transition_progress) +
                        self.target_weights[key] * transition_progress
                )

        for key, value in current_weights.items():
            setattr(self.loss_fn, key, value)

    def _cosine_schedule(self, epoch):
        progress = epoch / self.total_epochs
        cos_factor = 0.5 * (1 + np.cos(np.pi * progress))

        self.loss_fn.bce_weight = self.target_weights['bce_weight'] * (cos_factor * 0.7 + 0.3)
        self.loss_fn.focal_weight = self.target_weights['focal_weight'] * (cos_factor * 0.6 + 0.4)

        self.loss_fn.dice_weight = self.target_weights['dice_weight'] * ((1 - cos_factor) * 0.5 + 1.0)

        self.loss_fn.continuity_weight = self.target_weights['continuity_weight'] * ((1 - cos_factor) * 0.7 + 0.5)
        self.loss_fn.direction_weight = self.target_weights['direction_weight'] * ((1 - cos_factor) * 0.7 + 0.5)
        self.loss_fn.gradient_weight = self.target_weights['gradient_weight'] * ((1 - cos_factor) * 0.8 + 0.6)

    def _linear_schedule(self, epoch):
        progress = epoch / self.total_epochs

        self.loss_fn.bce_weight = self.target_weights['bce_weight'] * (1 - progress * 0.5)
        self.loss_fn.focal_weight = self.target_weights['focal_weight'] * (1 - progress * 0.5)

        self.loss_fn.dice_weight = self.target_weights['dice_weight'] * (1 + progress * 0.5)

        self.loss_fn.continuity_weight = self.target_weights['continuity_weight'] * (1 + progress)
        self.loss_fn.direction_weight = self.target_weights['direction_weight'] * (1 + progress)
        self.loss_fn.gradient_weight = self.target_weights['gradient_weight'] * (1 + progress * 0.8)

    def _step_schedule(self, epoch):
        if epoch < self.total_epochs * 0.3:
            weights = self.initial_weights
        elif epoch < self.total_epochs * 0.7:
            weights = self.mid_weights
        else:
            weights = self.target_weights

        for key, value in weights.items():
            setattr(self.loss_fn, key, value)

    def _apply_current_weights(self):
        current_weights = {
            'bce_weight': self.loss_fn.bce_weight,
            'focal_weight': self.loss_fn.focal_weight,
            'dice_weight': self.loss_fn.dice_weight,
            'tversky_weight': self.loss_fn.tversky_weight,
            'continuity_weight': self.loss_fn.continuity_weight,
            'direction_weight': self.loss_fn.direction_weight,
            'gradient_weight': self.loss_fn.gradient_weight
        }

        self._normalize_weights(current_weights)

        for key, value in current_weights.items():
            setattr(self.loss_fn, key, value)

    def get_current_weights(self):
        return {
            'bce_weight': self.loss_fn.bce_weight,
            'focal_weight': self.loss_fn.focal_weight,
            'dice_weight': self.loss_fn.dice_weight,
            'tversky_weight': self.loss_fn.tversky_weight,
            'continuity_weight': self.loss_fn.continuity_weight,
            'direction_weight': self.loss_fn.direction_weight,
            'gradient_weight': self.loss_fn.gradient_weight
        }


def create_edge_loss(loss_type='unified', **kwargs):
    if loss_type == 'unified':
        return UnifiedEdgeDetectionLoss(**kwargs)
    elif loss_type == 'bce':
        return BalancedBCELoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'continuity':
        return EdgeContinuityLoss(**kwargs)
    elif loss_type == 'direction':
        return EdgeDirectionConsistencyLoss(**kwargs)
    elif loss_type == 'gradient':
        return GradientLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def create_loss_with_scheduler(total_epochs=100, schedule_type='dynamic', **loss_kwargs):
    loss_fn = UnifiedEdgeDetectionLoss(**loss_kwargs)
    scheduler = LossScheduler(loss_fn, schedule_type=schedule_type, total_epochs=total_epochs)
    return loss_fn, scheduler
