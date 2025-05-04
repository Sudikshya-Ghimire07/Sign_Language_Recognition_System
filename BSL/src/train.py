'''
USAGE:
python train.py --epochs 10
'''

import os
import pandas as pd
import joblib
import numpy as np
import torch
import random
import albumentations as A
import matplotlib.pyplot as plt
import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import time
import cv2
import cnn_models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from torch.utils.data import Dataset, DataLoader

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
args = vars(parser.parse_args())

# Set random seed
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(SEED)

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Computation device: {device}")

# Load data
df = pd.read_csv('../input/data.csv')
X = df.image_path.values
y = df.target.values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=SEED)

print(f"Training on {len(xtrain)} images")
print(f"Validating on {len(xtest)} images")

# Dataset class
class ASLImageDataset(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels
        self.aug = A.Compose([A.Resize(224, 224)])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        image = cv2.imread(self.X[i])
        image = self.aug(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)

# Dataloaders
train_data = ASLImageDataset(xtrain, ytrain)
test_data = ASLImageDataset(xtest, ytest)
trainloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


# Model
model = cnn_models.CustomCNN().to(device)
print(model)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_params:,} total parameters.")
print(f"{trainable_params:,} training parameters.")

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Validation function
def validate(model, dataloader):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == target).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_accuracy = 100. * running_correct / len(dataloader.dataset)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
    return val_loss, val_accuracy, all_preds, all_targets

# Training function
def fit(model, dataloader):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0

    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(dataloader.dataset)
    train_accuracy = 100. * running_correct / len(dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    return train_loss, train_accuracy

# Main training loop
if __name__ == '__main__':
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    all_preds, all_targets = [], []

    start = time.time()
    for epoch in range(args['epochs']):
        print(f"\nEpoch {epoch+1}/{args['epochs']}")
        tr_loss, tr_acc = fit(model, trainloader)
        vl_loss, vl_acc, preds, targets = validate(model, testloader)

        train_loss.append(tr_loss)
        train_accuracy.append(tr_acc)
        val_loss.append(vl_loss)
        val_accuracy.append(vl_acc)

        all_preds = preds
        all_targets = targets
    end = time.time()
    print(f"\nTraining completed in {(end - start) / 60:.2f} minutes")

    os.makedirs('../outputs', exist_ok=True)
    os.makedirs('../chart', exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy, label='Train Acc', color='green')
    plt.plot(val_accuracy, label='Val Acc', color='blue')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../chart/accuracy.png')
    plt.show()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss', color='orange')
    plt.plot(val_loss, label='Val Loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../chart/loss.png')
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("../chart/confusion_matrix.png")
    plt.show()

    # Per-class metrics
    class_names = np.unique(y)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=class_names
    )

    x = np.arange(len(class_names))
    width = 0.25

    # Precision, Recall, F1-score per class
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width, label='F1-score', color='salmon')
    plt.xticks(ticks=x, labels=class_names, rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-score per Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../chart/precision_recall_f1_per_class.png')
    plt.show()

    # Support per class
    plt.figure(figsize=(12, 5))
    plt.bar(class_names, support, color='purple')
    plt.xticks(rotation=90)
    plt.ylabel("Number of Samples")
    plt.title("Support (Samples per Class)")
    plt.tight_layout()
    plt.savefig("../chart/class_support.png")
    plt.show()

    # Per-class accuracy
    cm = confusion_matrix(all_targets, all_preds, labels=class_names)
    per_class_acc = cm.diagonal() / support

    plt.figure(figsize=(12, 5))
    plt.bar(class_names, per_class_acc, color='orange')
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig("../chart/per_class_accuracy.png")
    plt.show()

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), '../outputs/model.pth')
