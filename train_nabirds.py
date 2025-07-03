import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Config
DATA_DIR = './nabirds'
IMG_DIR = os.path.join(DATA_DIR, 'images')
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_PATH = 'nabirds_model.pth'

# Metadata
images_df = pd.read_csv(os.path.join(DATA_DIR, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
labels_df = pd.read_csv(os.path.join(DATA_DIR, 'image_class_labels.txt'), sep=' ', names=['img_id', 'label'])
split_df = pd.read_csv(os.path.join(DATA_DIR, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_train'])

df = images_df.merge(labels_df, on='img_id').merge(split_df, on='img_id')
df['filepath'] = df['filepath'].apply(lambda x: os.path.join(IMG_DIR, x))
1
unique_labels = sorted(df['label'].unique())
label_map = {original: new for new, original in enumerate(unique_labels)}
df['label'] = df['label'].map(label_map)
NUM_CLASSES = len(unique_labels)


# Dataset class
class NABirdsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        label = row['label']
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_df = df[df['is_train'] == 1]
val_df = df[df['is_train'] == 0]

train_dataset = NABirdsDataset(train_df, transform=transform)
val_dataset = NABirdsDataset(val_df, transform=transform)

# ⛔ Don't create DataLoaders globally if using multiprocessing on Windows
def get_dataloaders():
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader

# Wrap all training logic here
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_dataloaders()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", unit="batch")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"[Epoch {epoch+1}] Train Loss: {running_loss:.4f}, Validation Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[✅] Model saved to {MODEL_PATH}")

# Required for Windows multiprocessing
if __name__ == '__main__':
    train()
