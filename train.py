"""
train.py — ASL Gesture Classifier (MobileNetV2, Transfer Learning)
PyTorch implementation — two-phase training: feature extraction → fine-tuning
29 classes: A-Z + SPACE + DELETE + NOTHING
"""

import os
import json
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR         = #Add your own
IMG_SIZE         = 128
BATCH_SIZE       = 32
NUM_CLASSES      = 29
MODEL_SAVE       = "model.pth"
LABELS_SAVE      = "labels.json"
VALIDATION_SPLIT = 0.15
SEED             = 42

# Use 20% of training data per epoch — sees full dataset over 5 epochs
# Keeps each epoch fast while still training on everything
SUBSET_RATIO     = 0.20

# Phase 1 — feature extraction (base frozen)
P1_EPOCHS        = 12
P1_LR            = 1e-3

# Phase 2 — fine-tuning (partial unfreeze)
P2_EPOCHS        = 15
P2_LR            = 1e-4
UNFREEZE_FROM    = 14

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


# ── Data ──────────────────────────────────────────────────────────────────────
def build_dataloaders(data_dir):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.4, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    total      = len(full_dataset)
    val_size   = int(total * VALIDATION_SPLIT)
    train_size = total - val_size
    generator  = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Subsample training set for faster epochs
    subset_size    = int(train_size * SUBSET_RATIO)
    subset_indices = np.random.choice(train_size, subset_size, replace=False)

    # Apply val-only transforms to validation subset
    val_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(subset_indices),
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, full_dataset.class_to_idx


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def unfreeze_top_layers(model, unfreeze_from: int):
    features = list(model.features.children())
    for block in features[unfreeze_from:]:
        for param in block.parameters():
            param.requires_grad = True
    unfrozen = sum(p.requires_grad for p in model.parameters())
    total    = sum(1 for _ in model.parameters())
    print(f"Unfrozen params: {unfrozen} / {total}")


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              num_epochs, phase_name, patience=4):
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve   = 0
    scaler       = torch.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"[{phase_name}] Epoch {epoch}/{num_epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_SAVE)
            print(f"  ✓ Best model saved (val_acc: {best_val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_weights)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== ASL Classifier Training (PyTorch) ===\n")

    train_loader, val_loader, class_to_idx = build_dataloaders(DATA_DIR)
    num_classes = len(class_to_idx)
    print(f"Classes found: {num_classes}")
    assert num_classes == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, got {num_classes}. "
        "Check your dataset directory for missing/extra folders."
    )

    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    with open(LABELS_SAVE, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Labels saved → {LABELS_SAVE}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model     = build_model(num_classes)

    # ── Phase 1: Feature Extraction ───────────────────────────────────────────
    print("--- Phase 1: Feature Extraction (base frozen) ---")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=P1_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    model = run_phase(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, P1_EPOCHS, "Phase1")

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────────────────
    print("\n--- Phase 2: Fine-Tuning (partial unfreeze) ---")
    unfreeze_top_layers(model, unfreeze_from=UNFREEZE_FROM)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=P2_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    model = run_phase(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, P2_EPOCHS, "Phase2")

    print(f"\n✓ Training complete. Model saved → {MODEL_SAVE}")
    print(f"✓ Labels saved    → {LABELS_SAVE}")


if __name__ == "__main__":
    main()
