# -*- coding: utf-8 -*-
"""
UNSIR / Noise Attack (fixed for macOS & local runs)

- No os.chdir; uses pathlib for stable paths
- No multiprocessing: DataLoader(num_workers=0)
- Consistent model: DenseNet-121 everywhere
- No Colab magics
"""

from pathlib import Path
import os
import random
import tarfile
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.models import densenet121  # keep architecture consistent


# ──────────────────────────────────────────────────────────────
# Paths & device
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# macOS: prefer MPS (Apple Silicon) if present; else CUDA; else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir    : {DATA_DIR}")
print(f"Models dir  : {MODELS_DIR}")
print(f"Device      : {device}")


# ──────────────────────────────────────────────────────────────
# Reproducibility (basic)
# ──────────────────────────────────────────────────────────────
seed = random.randint(0, 2**32 - 1)
random.seed(seed)
np.random.seed(seed % (2**32 - 1))
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)
print(f"Using random seed: {seed}")


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def training_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    return loss


@torch.no_grad()
def validation_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return {"Loss": loss.detach(), "Acc": acc}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in loader]
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg["lr"]


def fit_one_cycle(
    epochs,
    max_lr,
    model,
    train_loader,
    val_loader,
    device,
    weight_decay=0.0,
    grad_clip=None,
    opt_func=torch.optim.SGD,
):
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))

        # Validation
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"last_lr: {result['lrs'][-1]:.5f}, "
            f"train_loss: {result['train_loss']:.4f}, "
            f"val_loss: {result['Loss']:.4f}, "
            f"val_acc: {result['Acc']:.4f}"
        )
        history.append(result)
        sched.step(result["Loss"])
    return history


# Optional: a simple learnable noise module (kept for parity with your code)
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


# ──────────────────────────────────────────────────────────────
# Data download & setup
# ──────────────────────────────────────────────────────────────
def prepare_cifar10(data_dir: Path, project_root: Path):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    print("Downloading CIFAR-10 (if needed)…")
    download_url(url, str(project_root))  # saves under project root
    tgz_path = project_root / "cifar10.tgz"
    print("Extracting…")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir)  # -> data/cifar10
    cifar_dir = data_dir / "cifar10"
    print("Data ready at:", cifar_dir)
    return cifar_dir


def build_loaders(cifar_dir: Path, batch_size=256):
    transform_train = tt.Compose(
        [
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = tt.Compose(
        [
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_ds = ImageFolder(cifar_dir / "train", transform_train)
    valid_ds = ImageFolder(cifar_dir / "test", transform_test)

    # IMPORTANT: no worker processes on macOS (per your request)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False
    )
    class_names = sorted(os.listdir(cifar_dir / "train"))
    print("Classes:", class_names)
    return train_ds, valid_ds, train_dl, valid_dl, class_names


# ──────────────────────────────────────────────────────────────
# Main training / evaluation pipeline
# ──────────────────────────────────────────────────────────────
def main():
    # Prepare data
    cifar_dir = prepare_cifar10(DATA_DIR, PROJECT_ROOT)
    train_ds, valid_ds, train_dl, valid_dl, class_names = build_loaders(cifar_dir)

    # Model
    model = densenet121(num_classes=10).to(device)

    # Training config
    epochs = 1          # increase to ~80 for a full run
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    # Train
    print("\n=== Training DenseNet-121 on CIFAR-10 ===")
    fit_one_cycle(
        epochs,
        max_lr,
        model,
        train_dl,
        valid_dl,
        device,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
        opt_func=opt_func,
    )

    # Save & reload
    base_model_path = MODELS_DIR / "densenet121_CIFAR10_ALL_CLASSES.pt"
    torch.save(model.state_dict(), base_model_path)
    print("Saved model to:", base_model_path)

    model.load_state_dict(torch.load(base_model_path, map_location=device))

    # Evaluate
    print("\n=== Evaluation ===")
    train_res = evaluate(model, train_dl, device)
    print(f"Train Acc: {train_res['Acc']*100:.2f}% | Train Loss: {train_res['Loss']:.4f}")
    val_res = evaluate(model, valid_dl, device)
    print(f"Test  Acc: {val_res['Acc']*100:.2f}% | Test  Loss: {val_res['Loss']:.4f}")

    # ──────────────────────────────────────────────────────────
    # Unlearning scaffolding (kept minimal & consistent)
    # If you later enable the noise/impair/repair steps, keep model=DenseNet-121
    # and reuse the same loaders. Also keep num_workers=0.
    # ──────────────────────────────────────────────────────────

    # Example: build retain/forget validation sets (class 0 forgotten)
    classes_to_forget = [0]  # "airplane" in CIFAR-10 by default ordering
    num_classes = 10

    # Build classwise lists
    classwise_train = {i: [] for i in range(num_classes)}
    for img, label in train_ds:
        classwise_train[label].append((img, label))
    classwise_test = {i: [] for i in range(num_classes)}
    for img, label in valid_ds:
        classwise_test[label].append((img, label))

    # Retain samples
    num_samples_per_class = 1000
    retain_samples = []
    for cls in range(num_classes):
        if cls not in classes_to_forget:
            retain_samples += classwise_train[cls][:num_samples_per_class]

    # Validation: retain vs forget
    retain_valid = []
    for cls in range(num_classes):
        if cls not in classes_to_forget:
            retain_valid.extend(classwise_test[cls])
    forget_valid = []
    for cls in range(num_classes):
        if cls in classes_to_forget:
            forget_valid.extend(classwise_test[cls])

    # DataLoaders (no workers)
    forget_valid_dl = DataLoader(forget_valid, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    retain_valid_dl = DataLoader(retain_valid, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    print("\n=== Baseline (before unlearning) ===")
    res_forget = evaluate(model, forget_valid_dl, device)
    print(f"Forget-class Acc: {res_forget['Acc']*100:.2f}% | Loss: {res_forget['Loss']:.4f}")
    res_retain = evaluate(model, retain_valid_dl, device)
    print(f"Retain-classes Acc: {res_retain['Acc']*100:.2f}% | Loss: {res_retain['Loss']:.4f}")

    # If/when you implement impair/repair, keep architecture (DenseNet-121) and save here:
    impair_repair_path = MODELS_DIR / "impair_repair_weights.pt"
    torch.save(model.state_dict(), impair_repair_path)
    print("Saved (placeholder) impair/repair weights to:", impair_repair_path)

    # Reload to demonstrate consistency
    model.load_state_dict(torch.load(impair_repair_path, map_location=device))
    print("\nReloaded impair/repair weights for evaluation:")
    res_forget = evaluate(model, forget_valid_dl, device)
    print(f"Forget-class Acc: {res_forget['Acc']*100:.2f}% | Loss: {res_forget['Loss']:.4f}")
    res_retain = evaluate(model, retain_valid_dl, device)
    print(f"Retain-classes Acc: {res_retain['Acc']*100:.2f}% | Loss: {res_retain['Loss']:.4f}")


if __name__ == "__main__":
    # No multiprocessing needed (you asked for num_workers=0 everywhere)
    main()
