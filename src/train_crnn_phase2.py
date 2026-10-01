import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# --- Phase 2 Configuration ---
IMG_DIR = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-images"
CSV_FILE = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-labels.csv"
WEIGHTS_PATH = "weights/crnn_ocr_best.pt"

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001  # 10x lower learning rate to preserve pre-trained shapes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_INT = {char: idx + 1 for idx, char in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB) + 1  # +1 for CTC blank token

class PlateDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file, header=None, names=["filename", "label"])
        self.df.dropna(subset=["filename", "label"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["filename"]))
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Safe fallback if a real image is corrupted or missing
        if img is None:
            img = np.zeros((32, 128), dtype=np.float32)
        else:
            img = cv2.resize(img, (128, 32))
            img = (img / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]

        tensor_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        label_str = str(row["label"]).replace(" ", "").upper()
        target = [CHAR_TO_INT[c] for c in label_str if c in CHAR_TO_INT]
        target_len = len(target)
        
        return tensor_img, torch.tensor(target, dtype=torch.long), torch.tensor(target_len, dtype=torch.long)

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.stack(target_lengths)
    
    input_lengths = torch.full(size=(len(batch),), fill_value=32, dtype=torch.long)
    return images, targets, input_lengths, target_lengths

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.GRU(512, 128, bidirectional=True, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).reshape(b, w, c * h)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits.permute(1, 0, 2).log_softmax(2)

# --- Training Setup ---
dataset = PlateDataset(CSV_FILE, IMG_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = CRNN(NUM_CLASSES).to(DEVICE)

# --- LOAD PHASE 1 SYNTHETIC PRE-TRAINED WEIGHTS ---
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print(f"[*] Successfully loaded pre-trained baseline from {WEIGHTS_PATH}")
else:
    print(f"[!] Critical Error: {WEIGHTS_PATH} not found. You must run Phase 1 first.")
    exit()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

print(f"[*] Starting Phase 2 Fine-Tuning on: {DEVICE}")
print(f"[*] Real Dataset samples: {len(dataset)} | Batches per epoch: {len(dataloader)}\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(dataloader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Batch logging for real-time progress
        if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(dataloader)}] | Current Batch Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"---> Epoch {epoch+1}/{EPOCHS} Finished | Real CCTV Average Loss: {avg_loss:.4f}\n")

# Overwrite weights with fine-tuned parameters
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"[*] Phase 2 Fine-Tuning Complete! Production weights saved to {WEIGHTS_PATH}")