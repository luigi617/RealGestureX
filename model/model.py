import torch
try:
    from u2net.u2net import U2NET # Ensure U^2-Net is correctly installed
except:
    from model.u2net.u2net import U2NET

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the U^2-Net model
sal_model = U2NET(3, 1).to(device)

# Load pre-trained weights
sal_model.load_state_dict(torch.load('u2net/u2net.pth', map_location=device))

# Set model to evaluation mode
sal_model.eval()

from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def generate_saliency_mask(image_path, model, device):
    """
    Generates a saliency mask for the given image using U^2-Net.
    
    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): Pre-trained saliency detection model.
        device (torch.device): Device to perform computation on.
    
    Returns:
        np.array: Saliency mask normalized between 0 and 1.
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = model(input_tensor)
        pred = d1[:, 0, :, :]
        pred = pred.cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # Normalize to [0,1]
        pred = np.uint8(pred * 255)
        pred = Image.fromarray(pred).resize(image.size, resample=Image.BILINEAR)
        pred = np.array(pred) / 255.0  # Normalize to [0,1]
    
    return pred


import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KonIQ10kDataset(Dataset):
    def __init__(self, images_dir, csv_file, saliency_model, device, transform=None):
        """
        Args:
            images_dir (str): Path to images.
            csv_file (str): Path to the CSV file with global scores.
            saliency_model (torch.nn.Module): Pre-trained saliency detection model.
            device (torch.device): Device to perform computation on.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.global_scores = pd.read_csv(csv_file)
        self.transform = transform
        self.saliency_model = saliency_model
        self.device = device

    def __len__(self):
        return len(self.global_scores)

    def __getitem__(self, idx):
        # Get image filename and global score
        img_name = self.global_scores.iloc[idx, 0]
        score = self.global_scores.iloc[idx, 1]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Generate saliency mask
        saliency_mask = generate_saliency_mask(img_path, self.saliency_model, self.device)
        saliency_mask = np.expand_dims(saliency_mask, axis=-1)  # Add channel dimension
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_np, mask=saliency_mask)
            image = augmented['image']
            saliency_mask = augmented['mask']
        
        # Convert mask to binary (threshold can be adjusted)
        saliency_mask = (saliency_mask > 0.5).float()
        
        return image, torch.tensor(score, dtype=torch.float32), saliency_mask.squeeze(0)


# Define image transformations
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

from torch.utils.data import DataLoader, random_split
import numpy as np

# Parameters
images_dir = '../koniq-10k/images/'  # Update with your actual path
csv_file = '../koniq-10k/annotations/koniq-10k.csv'  # Update with your actual path
batch_size = 16
validation_split = 0.2
shuffle_dataset = True
random_seed= 42

# Initialize the dataset
full_dataset = KonIQ10kDataset(images_dir, csv_file, saliency_model=sal_model, device=device, transform=get_transforms(train=True))

# Creating data indices for training and validation splits:
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
valid_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)


from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetIQA(nn.Module):
    def __init__(self, efficientnet_version='efficientnet-b0', pretrained=True):
        super(EfficientNetIQA, self).__init__()
        # Load EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained(efficientnet_version) if pretrained else EfficientNet.from_name(efficientnet_version)
        
        # Remove the classification head
        self.backbone._fc = nn.Identity()
        self.backbone._avg_pooling = nn.Identity()
        
        # Global Quality Assessment Head
        self.global_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Regression output
        )
        
        # Local Quality Assessment Head
        # We'll add convolutional layers to generate a quality map
        self.local_head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        # Pass through EfficientNet backbone
        features = self.backbone.extract_features(x)  # Shape: [B, 1280, H, W]
        
        # Global Quality
        # Adaptive pooling to get a fixed-size feature vector
        pooled = F.adaptive_avg_pool2d(features, (1,1)).view(features.size(0), -1)  # Shape: [B, 1280]
        global_quality = self.global_head(pooled).squeeze(1)  # Shape: [B]
        
        # Local Quality
        local_quality_map = self.local_head(features).squeeze(1)  # Shape: [B, H, W]
        local_quality_map = F.interpolate(local_quality_map, size=x.size(2), mode='bilinear', align_corners=False)  # Resize to input size
        
        return global_quality, local_quality_map


import torch
import torch.optim as optim

# Check for GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = EfficientNetIQA().to(device)

# Define loss functions
criterion_global = nn.MSELoss()
criterion_local = nn.BCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Optionally, define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# Parameters
num_epochs = 30
alpha = 1.0  # Weight for global loss
beta = 1.0   # Weight for local loss

# Initialize optimizer and other components (Assuming they are already defined)
# optimizer = ...
# criterion_global = ...
# criterion_local = ...

# Define scheduler after optimizer is defined
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

best_val_loss = float('inf')
patience_counter = 0
patience = 10

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Move model to device before starting training
model = EfficientNetIQA().to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_global_loss = 0.0
    running_local_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for images, scores, masks in loop:
        # Move data to the appropriate device
        images = images.to(device, non_blocking=True)
        scores = scores.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward pass
            outputs_global, outputs_local = model(images)
            
            # Compute losses
            loss_global = criterion_global(outputs_global, scores)
            loss_local = criterion_local(outputs_local, masks)
            loss = alpha * loss_global + beta * loss_local
        
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update running losses
        running_loss += loss.item()
        running_global_loss += loss_global.item()
        running_local_loss += loss_local.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), global_loss=loss_global.item(), local_loss=loss_local.item())
    
    # Validation after each epoch
    model.eval()
    val_loss = 0.0
    val_global_loss = 0.0
    val_local_loss = 0.0
    with torch.no_grad():
        for images, scores, masks in valid_loader:
            # Move data to the appropriate device
            images = images.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            outputs_global, outputs_local = model(images)
            
            # Compute losses
            loss_global = criterion_global(outputs_global, scores)
            loss_local = criterion_local(outputs_local, masks)
            loss = alpha * loss_global + beta * loss_local
            
            # Accumulate validation losses
            val_loss += loss.item()
            val_global_loss += loss_global.item()
            val_local_loss += loss_local.item()
    
    # Calculate average losses
    avg_train_loss = running_loss / len(train_loader)
    avg_train_global_loss = running_global_loss / len(train_loader)
    avg_train_local_loss = running_local_loss / len(train_loader)
    avg_val_loss = val_loss / len(valid_loader)
    avg_val_global_loss = val_global_loss / len(valid_loader)
    avg_val_local_loss = val_local_loss / len(valid_loader)
    
    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Global: {avg_train_global_loss:.4f} | Local: {avg_train_local_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f} | Global: {avg_val_global_loss:.4f} | Local: {avg_val_local_loss:.4f}")
    
    # Scheduler step after validation
    scheduler.step(avg_val_loss)
    
    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Optionally, save model checkpoints
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

# Load the best model after training
model = EfficientNetIQA().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
