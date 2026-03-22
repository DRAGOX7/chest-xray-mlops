# %%
import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Libraries loaded successfully!")
import os
import shutil
import random

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


train_dataset = datasets.ImageFolder(
    root=f"{TARGET_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{TARGET_DIR}/val",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{TARGET_DIR}/test",
    transform=val_test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_dataset.classes)

# %%
# %%
# 1. DATA TRANSFORMS
# We use Grayscale->3 channels because DenseNet expects RGB, but X-rays are B&W
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Re-create loaders with these new transforms
# (Make sure DATA_DIR is defined above this!)
train_dataset = datasets.ImageFolder(os.path.join(TARGET_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(TARGET_DIR, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("✅ Data Loaders updated with DenseNet transforms")
# %%
# 2. HELPER FUNCTIONS
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total
# %%
# 3. MODEL & TRAINING LOOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DenseNet121
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
# Change classifier for 2 classes (Normal vs Abnormal)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(device)

# Define Optimizer (Missing in your snippet!)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"✅ DenseNet-121 loaded on {device}")

# Training Config
EPOCHS = 20
patience = 5
counter = 0
best_val_acc = 0.0
BEST_MODEL_PATH = "best_densenet121.pth"

# START TRAINING
if __name__ == "__main__":
    print("🚀 Starting training...")
    
    for epoch in range(EPOCHS):
        # Run one training epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        
        # Run evaluation
        val_acc = evaluate(model, val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Early Stopping & Saving Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   🎉 New best model saved! (Acc: {val_acc:.4f})")
        else:
            counter += 1
            print(f"   ⏳ No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print("⛔ Early stopping triggered")
                break

    print("✅ Training Complete.")