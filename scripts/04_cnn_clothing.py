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

# Configuration
DATA_DIR = r'E:\Projects\E-Commerce DM\data\raw\Clothes_Dataset'
MODEL_DIR = r'E:\Projects\E-Commerce DM\models'
OUTPUT_DIR = r'E:\Projects\E-Commerce DM\data\generated'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 20  # Keeping it low for fast training demo
IMG_SIZE = 64  # Resize images to 64x64 for speed
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# 1. Data Loading & Preprocessing
print("\n⏳ Loading dataset and applying transformations...")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    classes = dataset.classes
    print(f"✅ Found {len(dataset)} images across {len(classes)} classes:")
    print("   " + ", ".join(classes))
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Split 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📦 Train set: {train_size} images | Test set: {test_size} images")

# 2. Build the CNN Model
class ClothingCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClothingCNN, self).__init__()
        # Input: 3 x 64 x 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 8 * 8, 128),
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

# 3. Training Loop
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
start_time = time.time()

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
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"  Epoch {epoch+1}/{EPOCHS} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

print(f"✅ Training completed in {(time.time() - start_time):.1f} seconds")

# 4. Evaluation
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

# 5. Save Results
print("\n📝 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# Plot Confusion Matrix
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

# Save Model
model_path = os.path.join(MODEL_DIR, 'cnn_clothing.pth')
torch.save(model.state_dict(), model_path)
print(f"💾 Model weights saved to: {model_path}")
print("=" * 60)
print("  DAY 10 COMPLETED SUCCESSFULLY!")
print("=" * 60)
