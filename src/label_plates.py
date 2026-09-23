import os
import csv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- CONFIGURATION ---
CROPS_FOLDER = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates_new_1"
OUTPUT_CSV = r"C:\Users\User\Desktop\ANPR_project\Number plates\plates-labels.csv"
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

class PlateLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANPR Plate Labeler")
        self.root.geometry("520x360")
        self.root.resizable(False, False)

        # Load existing labels to resume progress
        self.labeled = set()
        if os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        self.labeled.add(row[0])

        # Scan for images
        if not os.path.exists(CROPS_FOLDER):
            messagebox.showerror("Error", f"Folder not found:\n{CROPS_FOLDER}")
            root.destroy()
            return

        all_files = [f for f in os.listdir(CROPS_FOLDER) if f.lower().endswith(VALID_EXTS)]
        self.image_files = [f for f in all_files if f not in self.labeled]
        self.total_count = len(all_files)
        self.current_idx = 0

        # UI Elements
        self.status_label = tk.Label(root, text="", font=("Segoe UI", 11, "bold"), fg="#2b5797")
        self.status_label.pack(pady=(12, 6))

        self.img_label = tk.Label(root, bg="#1e1e1e", width=460, height=150)
        self.img_label.pack(pady=6)

        entry_frame = tk.Frame(root)
        entry_frame.pack(pady=10)

        tk.Label(entry_frame, text="Plate Text: ", font=("Segoe UI", 11)).pack(side=tk.LEFT)
        self.entry = tk.Entry(entry_frame, font=("Segoe UI", 14, "bold"), width=16, justify='center')
        self.entry.pack(side=tk.LEFT, padx=6)
        self.entry.bind("<Return>", self.save_and_next)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=4)

        tk.Button(btn_frame, text="Skip (Enter)", width=12, command=self.skip_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save & Next", width=12, bg="#0078d4", fg="white", command=self.save_and_next).pack(side=tk.LEFT, padx=5)

        self.load_image()

    def load_image(self):
        if self.current_idx >= len(self.image_files):
            messagebox.showinfo("Finished", "All images in the folder have been labeled!")
            self.root.destroy()
            return

        img_name = self.image_files[self.current_idx]
        progress_num = self.total_count - len(self.image_files) + self.current_idx + 1
        self.status_label.config(text=f"[{progress_num}/{self.total_count}]  {img_name}")

        img_path = os.path.join(CROPS_FOLDER, img_name)
        try:
            pil_img = Image.open(img_path)
            pil_img = pil_img.resize((450, 140), Image.Resampling.BILINEAR)
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.img_label.config(image=self.tk_img)
        except Exception as e:
            print(f"Failed to load image {img_name}: {e}")

        self.entry.delete(0, tk.END)
        self.entry.focus_set()

    def save_and_next(self, event=None):
        raw_text = self.entry.get().strip().upper()
        clean_text = raw_text.replace(" ", "").replace("-", "")

        if clean_text:
            img_name = self.image_files[self.current_idx]
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([img_name, clean_text])
            self.labeled.add(img_name)

        self.current_idx += 1
        self.load_image()

    def skip_image(self):
        self.current_idx += 1
        self.load_image()

if __name__ == "__main__":
    app_root = tk.Tk()
    app = PlateLabelerApp(app_root)
    app_root.mainloop()