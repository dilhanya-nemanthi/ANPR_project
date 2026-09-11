#Training model with numberplates

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
CROPS_DIR = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-images"
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_INT = {char: idx + 1 for idx, char in enumerate(VOCAB)}  # 0 is reserved for CTC blank token
INT_TO_CHAR = {idx + 1: char for idx, char in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1  # +1 for CTC blank
IMG_HEIGHT = 32
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. DATASET CLASS ---
class PlateDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
    
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['filename']))
        
        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = (img / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)

        # Convert label text to numerical target
        label_text = str(row['label'])
        target = [CHAR_TO_INT[c] for c in label_text if c in CHAR_TO_INT]
        target_len = len(target)

        return img_tensor, torch.tensor(target, dtype=torch.long), target_len

def collate_fn(batch):
    imgs, targets, target_lens = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    flat_targets = torch.cat(targets)
    target_lens = torch.tensor(target_lens, dtype=torch.long)
    return imgs, flat_targets, target_lens

# --- 2. CRNN NETWORK ARCHITECTURE ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (32, 16, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (64, 8, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # (128, 4, 32)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1))   # (256, 2, 32)
        )
        
        # RNN Sequence Reader (Bidirectional GRU)
        self.rnn = nn.GRU(256 * 2, 128, bidirectional=True, batch_first=True, num_layers=2)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # (B, C, H, W)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2)  # (B, W, C, H)
        features = features.reshape(b, w, c * h) # (B, W, 512)
        
        rnn_out, _ = self.rnn(features)          # (B, W, 256)
        logits = self.fc(rnn_out)                # (B, W, num_classes)
        
        # CTCLoss expects (Time_steps, Batch, Num_classes) in log-softmax
        return logits.permute(1, 0, 2).log_softmax(2)

# --- 3. TRAINING LOOP ---
def train():
    train_dataset = PlateDataset("train_labels.csv", CROPS_DIR)
    val_dataset = PlateDataset("val_labels.csv", CROPS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on device: {DEVICE}")
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for imgs, targets, target_lens in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)  # (T, B, C)
            input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long, device=DEVICE)

            loss = criterion(preds, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, targets, target_lens in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(imgs)
                input_lens = torch.full((imgs.size(0),), preds.size(0), dtype=torch.long, device=DEVICE)
                loss = criterion(preds, targets, input_lens, target_lens)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/crnn_ocr_best.pt")

    print(f"\nTraining Complete! Best model saved to weights/crnn_ocr_best.pt (Val Loss: {best_loss:.4f})")

if __name__ == "__main__":
    train()