import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw
from typing import List
from torch import tensor
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import cv2


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)
    

class Hourglass(nn.Module):
    def __init__(self, depth, num_features):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.upper_branch = nn.ModuleList([ResidualBlock(num_features, num_features) for _ in range(depth)])
        self.lower_branch = nn.ModuleList([ResidualBlock(num_features, num_features) for _ in range(depth)])
        self.skip_branch = nn.ModuleList([ResidualBlock(num_features, num_features) for _ in range(depth)])

    def forward(self, x, level=0):
        if level == self.depth - 1:
            return self.lower_branch[level](x)

        up1 = self.upper_branch[level](x)
        low1 = self.downsample(up1)
        low2 = self.forward(low1, level + 1)
        low3 = self.lower_branch[level](low2)
        up2 = self.upsample(low3)

        skip = self.skip_branch[level](x)
        return up2 + skip
    

class StackedHourglass(nn.Module):
    def __init__(self, num_stacks=2, num_features=256, num_keypoints=10):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_features = num_features

        self.conv1 = nn.Conv2d(3, num_features // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features // 2)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(num_features // 2, num_features)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.hourglasses = nn.ModuleList([Hourglass(depth=4, num_features=num_features) for _ in range(num_stacks)])
        self.residuals = nn.ModuleList([ResidualBlock(num_features, num_features) for _ in range(num_stacks)])
        self.out_convs = nn.ModuleList([nn.Conv2d(num_features, num_keypoints, kernel_size=1) for _ in range(num_stacks)])
        self.heatmap_convs = nn.ModuleList([nn.Conv2d(num_keypoints, num_features, kernel_size=1) for _ in range(num_stacks - 1)])
        self.intermediate_convs = nn.ModuleList([nn.Conv2d(num_features, num_features, kernel_size=1) for _ in range(num_stacks - 1)])
        self.intermediate_residuals = nn.ModuleList([ResidualBlock(num_features, num_features) for _ in range(num_stacks - 1)])

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.pool(x)

        for i in range(self.num_stacks):
            hg = self.hourglasses[i](x)
            res = self.residuals[i](hg)
            out = self.out_convs[i](res)
            out_upsampled = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
            outputs.append(out_upsampled)

            if i < self.num_stacks - 1:
                out_transformed = self.heatmap_convs[i](out)
                x = x + self.intermediate_convs[i](res) + out_transformed
                x = self.intermediate_residuals[i](x)

        return outputs[-1]

class FaceLandmarksModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = StackedHourglass(2, num_features=256, num_keypoints=5)
        self.criterion = nn.MSELoss()
        self.transform = transform
        self.photos = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-6)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

model_shg = torch.load('models/sgh_95_epoch.pth', weights_only=False)


def create_heatmap(size, landmark, sigma=2):
    """
    Создаёт один heatmap с гауссовым ядром вокруг точки.

    :param size: (height, width) — размер heatmap'а
    :param landmark:(x, y) — координаты точки
    :param sigma
    :return: heatmap массив
    """
    x, y = landmark
    h, w = size


    x = min(max(0, int(x)), w - 1)
    y = min(max(0, int(y)), h - 1)

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    heatmap = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))
    return heatmap


def landmarks_to_heatmaps(image_shape, landmarks, sigma=2):
    """
    Преобразует список из N точек в набор из N heatmap'ов.

    :param image_shape: исходный размер изображения (H, W)
    :param landmarks: список из N пар координат [(x1, y1), (x2, y2), ..., (xN, yN),]
    :param sigma:
    :return: массив heatmap'ов вида [N, H, W]
    """
    heatmaps = []

    for i in range(5):
        x, y = landmarks[i]
        hm = create_heatmap(image_shape, (x, y), sigma=sigma)
        heatmaps.append(hm)

    return np.array(heatmaps)


def visualize_heatmaps_on_image(image, num_keypoints=5):
    photo = image.convert('RGB').resize((224, 224))
    photo_tensor = transform(image=np.array(photo))['image'].unsqueeze(0)
    
    model_shg.eval()
    with torch.no_grad():
        outputs = model_shg(photo_tensor)
        heatmaps = outputs.squeeze(0).cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Heatmaps Visualization', fontsize=16)
    
    axes[0, 0].imshow(photo)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    for i in range(num_keypoints):
        row = i // 3
        col = i % 3
        
        if row == 0 and col == 0:
            continue
        heatmap = heatmaps[i]
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]

        img_array = np.array(photo) / 255.0
        overlay = 0.7 * heatmap_colored + 0.3 * img_array
        
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f'Keypoint {i+1} Heatmap')
        axes[row, col].axis('off')
    
    if num_keypoints < 5:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return heatmaps

def visualize_combined_heatmap(image, num_keypoints=5):
    photo = np.array(image.convert('RGB').resize((224, 224)))
    photo_tensor = transform(image=photo)['image'].unsqueeze(0)
    

    model_shg.eval()
    with torch.no_grad():
        outputs = model_shg(photo_tensor)
        heatmaps = outputs.squeeze(0).cpu().numpy()


    
    combined_heatmap = np.zeros((224, 224))
    for i in range(num_keypoints):
        heatmap_normalized = (heatmaps[i] - heatmaps[i].min()) / (heatmaps[i].max() - heatmaps[i].min() + 1e-8)
        combined_heatmap += heatmap_normalized
    
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
    
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(photo)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(combined_heatmap, cmap='jet')
    ax2.set_title('Combined Heatmap')
    ax2.axis('off')
    
    img_array = np.array(photo) / 255.0
    heatmap_colored = plt.cm.jet(combined_heatmap)[:, :, :3]
    overlay = 0.7 * heatmap_colored + 0.3 * img_array
    
    ax3.imshow(overlay)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return combined_heatmap

def extract_keypoints_from_heatmaps(heatmaps):
    keypoints = []
    for heatmap in heatmaps:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        keypoints.append((x, y))
    return keypoints

def draw_keypoints_on_image(image):
    photo = image.convert('RGB').resize((224 , 224))
    photo_tensor = transform(image=np.array(photo))['image'].unsqueeze(0)
    
    model_shg.eval()
    with torch.no_grad():
        outputs = model_shg(photo_tensor)
        heatmaps = outputs.squeeze(0).cpu().numpy()
    keypoints = extract_keypoints_from_heatmaps(heatmaps)
    
    draw = ImageDraw.Draw(photo)
    color = 'red'
    
    for i, (x, y) in enumerate(keypoints):
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline='white')
        draw.text((x + 8, y - 8), str(i+1), fill='white')
    
    return photo

def get_key_points(image):
    photo = image.convert('RGB').resize((224 , 224))
    photo_tensor = transform(image=np.array(photo))['image'].unsqueeze(0)
    
    model_shg.eval()
    with torch.no_grad():
        outputs = model_shg(photo_tensor)
        heatmaps = outputs.squeeze(0).cpu().numpy()
    keypoints = extract_keypoints_from_heatmaps(heatmaps)
    return keypoints

def aligned_image(image):
    image = image.resize((224, 224))
    kp = [np.array(lst) for lst in get_key_points(image)]

    src_points = np.array([
        kp[0],
        kp[1],
        ((kp[4] + kp[3]) // 2 + kp[2]) // 2
    ], dtype=np.float32)

    dst_points = np.array([
        [60, 80],
        [160, 80],
        [110, 130]
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src_points, dst_points)

    aligned_image = cv2.warpAffine(np.array(image), M, (224, 224))
    return Image.fromarray(aligned_image)
