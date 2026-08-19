import os
import pandas as pd
from sklearn.model_selection import train_test_split

CROPS_DIR = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-images"        # Path to your crops folder
CSV_PATH = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-labels.csv"

# 1. Load CSV
df = pd.read_csv(CSV_PATH, header=None, names=['filename', 'label'])

# 2. Filter out missing image files
df['exists'] = df['filename'].apply(lambda x: os.path.exists(os.path.join(CROPS_DIR, str(x))))
df_clean = df[df['exists']].drop(columns=['exists'])

# 3. Clean labels (ensure string, uppercase, no spaces)
df_clean['label'] = df_clean['label'].astype(str).str.strip().str.upper()
df_clean = df_clean[df_clean['label'] != '']

print(f"Valid entries found: {len(df_clean)}")

# 4. Split into Train (80%) and Validation (20%)
train_df, val_df = train_test_split(df_clean, test_size=0.2, random_state=42)

train_df.to_csv("train_labels.csv", index=False, header=False)
val_df.to_csv("val_labels.csv", index=False, header=False)

print(f"Saved: {len(train_df)} training samples -> train_labels.csv")
print(f"Saved: {len(val_df)} validation samples -> val_labels.csv")