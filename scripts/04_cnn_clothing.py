import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print('=' * 60)
print('  Day 10: Deep Learning CNN - Clothing Classification')
print('=' * 60)

DATA_DIR = r'E:\Projects\E-Commerce DM\data\raw\Clothes_Dataset'
MODEL_DIR = r'E:\Projects\E-Commerce DM\models'
OUTPUT_DIR = r'E:\Projects\E-Commerce DM\data\generated'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30
IMG_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

print("\n⏳ Loading dataset and applying transformations...")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    classes = full_dataset.classes
    print(f"✅ Found {len(full_dataset)} images across {len(classes)} classes:")
    print("   " + ", ".join(classes))
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.indices)

generator = torch.Generator().manual_seed(42)
n = len(full_dataset)
train_size = int(0.70 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size

indices = torch.randperm(n, generator=generator).tolist()
train_dataset = TransformedSubset(full_dataset, indices[:train_size], train_transform)
val_dataset = TransformedSubset(full_dataset, indices[train_size:train_size+val_size], eval_transform)
test_dataset = TransformedSubset(full_dataset, indices[train_size+val_size:], eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📦 Train: {train_size} | Validation: {val_size} | Test: {test_size}")

class ClothingCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClothingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = ClothingCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"\n🚀 Starting training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
start_time = time.time()

best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(MODEL_DIR, 'cnn_clothing_best.pth')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
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

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc = 100. * val_correct / val_total

    scheduler.step(val_loss)

    print(f"  Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  ⚡ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

print(f"✅ Training completed in {(time.time() - start_time):.1f} seconds")

model.load_state_dict(torch.load(best_model_path, weights_only=True))

print("\n📊 Evaluating model on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n📝 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('CNN Confusion Matrix - Clothing Dataset')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'cnn_confusion_matrix.png')
plt.savefig(cm_path)
print(f"📈 Confusion matrix saved to: {cm_path}")

model_path = os.path.join(MODEL_DIR, 'cnn_clothing.pth')
torch.save(model.state_dict(), model_path)
print(f"💾 Best model weights saved to: {best_model_path}")
print(f"💾 Final model weights saved to: {model_path}")
print("=" * 60)
print("  DAY 10 COMPLETED SUCCESSFULLY!")
print("=" * 60)
