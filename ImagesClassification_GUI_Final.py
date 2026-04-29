
import os
import threading
import time
import json
import urllib.request
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import ttk

# -------------------------
# Configuration
# -------------------------
IMAGES_FOLDER = "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TOP_K = 5  # number of top predictions to display

# Model input size & normalization (ImageNet)
INPUT_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ImageNet class labels URL (official PyTorch mapping)
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_FILE = "imagenet_labels.json"


# -------------------------
# Model utilities
# -------------------------
def load_labels():
    """Load the 1000 ImageNet class labels from a local cache or download them."""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    # Download and cache
    urllib.request.urlretrieve(LABELS_URL, LABELS_FILE)
    with open(LABELS_FILE, "r") as f:
        return json.load(f)


def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # shape [1, 3, H, W]
    return image, tensor


def predict_image(model, tensor, labels, top_k=TOP_K):
    """Return top-k predictions as list of (class_name, confidence%)."""
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = int(top_indices[0][i].item())
        prob = float(top_probs[0][i].item()) * 100
        class_name = labels[idx] if idx < len(labels) else f"class_{idx}"
        results.append((class_name, prob, idx))

    return results


# -------------------------
# Helper functions
# -------------------------
def format_duration(seconds):
    """Return H:MM:SS or MM:SS from seconds (int/float)."""
    if seconds is None:
        return "--:--"
    seconds = int(round(seconds))
    td = str(timedelta(seconds=seconds))
    parts = td.split(", ")
    return parts[-1]


def format_predictions(results):
    """Format top-k results into a readable multi-line string."""
    lines = []
    for rank, (name, conf, idx) in enumerate(results, start=1):
        lines.append(f"  {rank}. {name} — {conf:.1f}%  (idx {idx})")
    return "\n".join(lines)


# -------------------------
# UI: Modernized App
# -------------------------
class ModernClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image Classifier — 1000 Classes")
        self.geometry("1000x760")
        self.minsize(900, 600)

        # ttk style
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Colors & fonts
        self.primary = "#B8B8B8"  # teal
        self.accent = "#000000"  # purple
        self.surface = "#000000"  # very dark
        self.card = "#000000"
        self.text = "#B8B8B8"
        self.muted = "#B8B8B8"
        self.font_heading = ("Inter", 16, "bold")
        self.font_normal = ("Inter", 11)
        self.font_mono = ("Consolas", 10)

        # style configuration
        self.style.configure("TFrame", background=self.surface)
        self.style.configure("Card.TFrame", background=self.card, relief="flat")
        self.style.configure("TLabel", background=self.surface, foreground=self.text, font=self.font_normal)
        self.style.configure("Heading.TLabel", background=self.surface, foreground=self.text, font=self.font_heading)
        self.style.configure("Muted.TLabel", background=self.surface, foreground=self.muted, font=self.font_normal)
        self.style.configure("Mono.TLabel", background=self.surface, foreground=self.text, font=self.font_mono)
        self.style.configure("TButton", font=self.font_normal, padding=6)
        self.style.map("Accent.TButton", background=[("active", self.primary), ("!active", self.accent)])
        self.style.configure("Accent.TButton", foreground="#fff", background=self.accent)
        self.style.configure("Primary.TButton", foreground="#fff", background=self.primary)
        self.style.configure("Danger.TButton", foreground="#fff", background="#EF4444")
        self.style.configure("TEntry", foreground=self.text)
        self.style.configure("TProgressbar", troughcolor=self.card, background=self.primary)

        # Internal state
        self.model = None
        self.labels = None
        self.current_photo = None
        self.selected_folder = IMAGES_FOLDER

        # Build UI
        self._build_ui()

        # Load model + labels in background
        self._set_status("Loading model & labels...")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    # -------------------------
    # UI building
    # -------------------------
    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=(18, 12), style="TFrame")
        top.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(top, text="AI Image Classifier", style="Heading.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text="Modern UI • 1000 ImageNet Classes", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 0))

        # Main content frame
        main = ttk.Frame(self, padding=14, style="TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        # Left: preview + log
        left = ttk.Frame(main, style="TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # Preview card
        preview_card = ttk.Frame(left, style="Card.TFrame", padding=12)
        preview_card.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(preview_card, text="Preview", style="Muted.TLabel").pack(anchor=tk.W)
        self.preview_label = ttk.Label(preview_card, text="No image", anchor="center", style="TLabel")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(10, 6))

        # inference info card — now shows top prediction + confidence
        info_card = ttk.Frame(preview_card, style="Card.TFrame")
        info_card.pack(fill=tk.X, pady=(4, 4))
        self.info_text = tk.StringVar(value="Prediction: —")
        ttk.Label(info_card, textvariable=self.info_text, style="Mono.TLabel").pack(anchor=tk.W, padx=4, pady=6)

        # Log card
        log_card = ttk.Frame(left, style="Card.TFrame", padding=8)
        log_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(log_card, text="Activity Log", style="Muted.TLabel").pack(anchor=tk.W)
        self.log_box = scrolledtext.ScrolledText(
            log_card,
            height=14,
            bg=self.card,
            fg=self.text,
            insertbackground=self.text,
            wrap=tk.WORD,
            font=self.font_mono,
            relief=tk.FLAT,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        self.log_box.insert(tk.END, "Ready. Choose an image or folder to classify.\n")
        self.log_box.config(state=tk.DISABLED)

        # Right: controls
        right = ttk.Frame(main, width=340, style="TFrame")
        right.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.N)

        controls_card = ttk.Frame(right, style="Card.TFrame", padding=12)
        controls_card.pack(fill=tk.X)

        ttk.Label(controls_card, text="Controls", style="Muted.TLabel").pack(anchor=tk.W)

        # single image controls
        single = ttk.Frame(controls_card, style="Card.TFrame", padding=(0, 8))
        single.pack(fill=tk.X, pady=(8, 6))

        ttk.Label(single, text="Single Image", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(single, textvariable=self.path_var, width=40)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=(0, 6))
        self.path_entry.bind("<Return>", lambda e: self._start_single())

        browse_btn = ttk.Button(single, text="Browse", style="TButton", command=self._browse_file)
        browse_btn.pack(side=tk.LEFT, padx=(8, 0))

        classify_btn = ttk.Button(single, text="Classify", style="Accent.TButton", command=self._start_single)
        classify_btn.pack(side=tk.LEFT, padx=(8, 0))

        # separator
        ttk.Separator(controls_card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # folder controls
        folder = ttk.Frame(controls_card, style="Card.TFrame", padding=(0, 4))
        folder.pack(fill=tk.X, pady=(4, 6))
        ttk.Label(folder, text="Folder Classification", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))

        folder_row = ttk.Frame(folder, style="Card.TFrame")
        folder_row.pack(fill=tk.X)
        self.folder_label_var = tk.StringVar(value=f"(default) {IMAGES_FOLDER}")
        ttk.Label(folder_row, textvariable=self.folder_label_var, style="TLabel", wraplength=200).pack(side=tk.LEFT)

        choose_folder_btn = ttk.Button(folder_row, text="Choose...", style="TButton", command=self._choose_folder)
        choose_folder_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # Classify folder button
        self.classify_folder_btn = ttk.Button(controls_card, text="Classify Folder", style="Primary.TButton",
                                              command=self._start_folder)
        self.classify_folder_btn.pack(fill=tk.X, pady=(10, 6))

        # Progress area
        prog_card = ttk.Frame(controls_card, style="Card.TFrame", padding=8)
        prog_card.pack(fill=tk.X, pady=(6, 8))
        ttk.Label(prog_card, text="Progress", style="Muted.TLabel").pack(anchor=tk.W)
        self.progress = ttk.Progressbar(prog_card, orient="horizontal", length=260, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(6, 6))
        bottom_row = ttk.Frame(prog_card, style="Card.TFrame")
        bottom_row.pack(fill=tk.X)
        self.percent_var = tk.StringVar(value="0%")
        self.eta_var = tk.StringVar(value="ETA: --:--")
        ttk.Label(bottom_row, textvariable=self.percent_var, style="TLabel", width=7).pack(side=tk.LEFT)
        ttk.Label(bottom_row, textvariable=self.eta_var, style="TLabel").pack(side=tk.LEFT)

        # clear & status
        ttk.Separator(controls_card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        clear_btn = ttk.Button(controls_card, text="Clear Log", style="TButton", command=self._clear_log)
        clear_btn.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Idle")
        status_bar = ttk.Frame(self, style="TFrame")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_var, style="Muted.TLabel", padding=8).pack(side=tk.LEFT)

        # Bindings for hover effect
        for b in (classify_btn, self.classify_folder_btn, clear_btn, browse_btn, choose_folder_btn):
            b.bind("<Enter>", lambda e, btn=b: btn.configure(cursor="hand2"))
            b.bind("<Leave>", lambda e, btn=b: btn.configure(cursor=""))

    # -------------------------
    # Model loading
    # -------------------------
    def _load_model_thread(self):
        self.labels = load_labels()
        self.model = load_model()
        self._set_status("Model loaded")
        self._log(f"Model loaded successfully. {len(self.labels)} classes available.")

    # -------------------------
    # UI helpers
    # -------------------------
    def _log(self, text):
        def _append():
            self.log_box.config(state=tk.NORMAL)
            self.log_box.insert(tk.END, f"{text}\n")
            self.log_box.see(tk.END)
            self.log_box.config(state=tk.DISABLED)

        self.after(0, _append)

    def _set_status(self, text):
        self.after(0, lambda: self.status_var.set(text))

    def _set_info(self, top_result=None):
        if top_result:
            name, conf, idx = top_result
            text = f"Top: {name} — {conf:.1f}%  (idx {idx})"
        else:
            text = "Prediction: —"
        self.after(0, lambda: self.info_text.set(text))

    def _update_preview(self, pil_image):
        def _do():
            width = 360
            h = pil_image.height * width // pil_image.width
            im = pil_image.resize((width, h))
            self.current_photo = ImageTk.PhotoImage(im)
            self.preview_label.config(image=self.current_photo, text="")

        self.after(0, _do)

    def _set_progress(self, value, maximum=None):
        def _do():
            if maximum is not None:
                self.progress["maximum"] = maximum
            self.progress["value"] = value
            maxv = float(self.progress["maximum"]) if float(self.progress["maximum"]) > 0 else 1.0
            percent = int(round((float(value) / maxv) * 100))
            self.percent_var.set(f"{percent}%")

        self.after(0, _do)

    def _set_eta(self, secs):
        text = "ETA: " + (format_duration(secs) if secs is not None else "--:--")
        self.after(0, lambda: self.eta_var.set(text))

    def _set_busy(self, busy=True):
        def _do():
            state = "disabled" if busy else "normal"
            self.classify_folder_btn.config(state=state)
            self.path_entry.config(state=state)

        self.after(0, _do)

    # -------------------------
    # Actions: file dialogs
    # -------------------------
    def _browse_file(self):
        file = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if file:
            self.path_var.set(file)

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Select images folder")
        if folder:
            self.selected_folder = folder
            self.folder_label_var.set(folder)
            self._log(f"Selected folder: {folder}")

    # -------------------------
    # Single-image classify
    # -------------------------
    def _start_single(self):
        path = self.path_var.get().strip()
        if not path or not self.model:
            return
        threading.Thread(target=self._classify_single_thread, args=(path,), daemon=True).start()

    def _classify_single_thread(self, path):
        self._set_busy(True)
        self._set_status("Classifying image...")
        self._set_progress(0, maximum=1)
        self._set_eta(None)

        image, tensor = preprocess_image(path)
        self._update_preview(image)

        results = predict_image(self.model, tensor, self.labels)
        self._set_info(top_result=results[0])

        self._log(f"Image: {Path(path).name}")
        self._log(f"  Top-{TOP_K} predictions:")
        self._log(format_predictions(results))

        self._set_progress(1, maximum=1)
        self._set_eta(0)
        self._set_status("Idle")
        self._set_busy(False)

    # -------------------------
    # Folder classify
    # -------------------------
    def _start_folder(self):
        if not self.model:
            return
        folder = self.selected_folder or IMAGES_FOLDER
        threading.Thread(target=self._classify_folder_thread, args=(folder,), daemon=True).start()

    def _classify_folder_thread(self, folder):
        files = sorted(os.listdir(folder))
        images = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        total = len(images)
        if total == 0:
            self._log("No supported images found in the selected folder.")
            return

        self._set_busy(True)
        self._set_status(f"Classifying {total} images...")
        self._set_progress(0, maximum=total)
        self._set_eta(None)
        self._log(f"--- Start folder classification: {folder} ({total} images) ---")

        start_time = time.time()

        for i, fname in enumerate(images, start=1):
            path = os.path.join(folder, fname)
            image, tensor = preprocess_image(path)
            self._update_preview(image)

            results = predict_image(self.model, tensor, self.labels)
            top_name, top_conf, top_idx = results[0]
            self._set_info(top_result=results[0])
            self._log(f"{fname} → {top_name} ({top_conf:.1f}%, idx {top_idx})")

            # update progress & ETA
            self._set_progress(i, maximum=total)
            elapsed = time.time() - start_time
            avg = elapsed / i
            remaining = (total - i) * avg if i < total else 0
            self._set_eta(remaining)

        self._log("--- Folder classification complete ---")
        self._set_status("Idle")
        self._set_progress(total, maximum=total)
        self._set_eta(0)
        self.after(900, lambda: (self._set_progress(0, maximum=1), self._set_eta(None), self._set_busy(False)))

    # -------------------------
    # Utility: clear log
    # -------------------------
    def _clear_log(self):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.insert(tk.END, "Ready. Choose an image or folder to classify.\n")
        self.log_box.config(state=tk.DISABLED)
        self.preview_label.config(image="", text="No image")
        self.info_text.set("Prediction: —")
        self._set_progress(0, maximum=1)
        self._set_eta(None)
        self._log("Cleared")


# -------------------------
# Main entry
# -------------------------
def main():
    app = ModernClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
