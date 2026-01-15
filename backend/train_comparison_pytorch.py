#!/usr/bin/env python3
"""PyTorch training script for waste classification models."""

import json
import os
import time
import copy
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
from tqdm import tqdm

SEED = 307713
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
EPOCHS = 15

DATASET_FOLDER_NAME = "merged_dataset"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "comparison_results_pytorch")

ENABLE_EFFICIENTNET_FINETUNE = False


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def setup_cuda_optimizations():
    """Enable TF32 and cudnn optimizations for RTX 40xx series."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("[CUDA] TF32 and cudnn.benchmark ENABLED")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ============================================================================
# DATA LOADING
# ============================================================================
def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_dataset_path():
    """Get path to the merged dataset (must be created by download_and_merge.py)."""
    # Force use of local merged dataset
    local_path = os.path.join(os.getcwd(), DATASET_FOLDER_NAME)
    
    if not os.path.exists(local_path):
        # Check if we are in root and need to go to backend
        local_path_alt = os.path.join(os.getcwd(), "backend", DATASET_FOLDER_NAME)
        if os.path.exists(local_path_alt):
            local_path = local_path_alt
    
    if os.path.exists(local_path):
        print(f"[INFO] Using local merged dataset at: {local_path}")
        return local_path
        
    print(f"[ERROR] Merged dataset not found at: {local_path}")
    print("[ERROR] Please run 'download_and_merge.py' first to create the dataset!")
    raise FileNotFoundError("Dataset not found. Run download_and_merge.py first.")

def prepare_datasets():
    """Prepare datasets using the merged dataset folder."""
    # Get dataset path
    dataset_root = get_dataset_path()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load full dataset
    full_dataset = datasets.ImageFolder(dataset_root)
    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Classes detected: {class_names}")
    
    # 2. Creates Splits (70% train, 15% val, 15% test)
    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=SEED)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED)
    
    # Create Subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    # Apply Transforms
    train_dataset = TransformedDataset(train_dataset, get_transforms("train"))
    val_dataset = TransformedDataset(val_dataset, get_transforms("val"))
    test_dataset = TransformedDataset(test_dataset, get_transforms("val"))
    
    # Save mappings
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    with open(os.path.join(RESULTS_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "idx_to_class.json"), "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    
    # Also save to BASE_DIR
    with open(os.path.join(BASE_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    
    print(f"[INFO] Class mappings saved")
    print(f"[INFO] Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 3. Create DataLoaders
    loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': PREFETCH_FACTOR
    }
    
    dataloaders = {
        'train': DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        'val': DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        'test': DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    }
    
    # 4. Feature Extraction Loaders (NO augmentation, deterministic)
    print("[INFO] Creating feature loaders...")
    
    train_feat_ds = TransformedDataset(Subset(full_dataset, train_idx), get_transforms("val"))
    val_feat_ds = TransformedDataset(Subset(full_dataset, val_idx), get_transforms("val"))
    test_feat_ds = TransformedDataset(Subset(full_dataset, test_idx), get_transforms("val"))
    
    # Use fewer workers for feature extraction if strict sequential order is not required but speed is good
    feat_loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': False
    }
    
    feature_loaders = {
        'train': DataLoader(train_feat_ds, shuffle=False, **feat_loader_kwargs),
        'val': DataLoader(val_feat_ds, shuffle=False, **feat_loader_kwargs),
        'test': DataLoader(test_feat_ds, shuffle=False, **feat_loader_kwargs)
    }
    
    return dataloaders, feature_loaders, class_names

class TransformedDataset(Dataset):
    """Dataset wrapper that applies transforms."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

# ============================================================================
# MODELS
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_classifier(model, classifier_attr='classifier'):
    classifier = getattr(model, classifier_attr)
    for p in classifier.parameters():
        p.requires_grad = True

def build_model(model_name, num_classes, device):
    if model_name == "simple_cnn":
        model = SimpleCNN(num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        freeze_backbone(model)
        unfreeze_classifier(model, 'classifier')
        print(f"  [INFO] {model_name}: backbone FROZEN, training classifier only")
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        freeze_backbone(model)
        unfreeze_classifier(model, 'classifier')
        print(f"  [INFO] {model_name}: backbone FROZEN, training classifier only")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    return model

# ============================================================================
# TRAINING
# ============================================================================
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, model_name):
    since = time.time()
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))
    use_amp = device.type == "cuda"
    
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    total_batches = len(dataloaders['train']) + len(dataloaders['val'])
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch+1}/{num_epochs}', 
                    unit='batch', ncols=100, leave=True)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                if inputs.dim() == 4 and device.type == 'cuda':
                    inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.update(1)
                pbar.set_postfix({'phase': phase, 'loss': f'{loss.item():.3f}'})
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        
        pbar.close()
        epoch_time = time.time() - epoch_start
        remaining = (num_epochs - epoch - 1) * epoch_time
        print(f'  ✓ train: {history["train_acc"][-1]:.4f} | val: {history["val_acc"][-1]:.4f} | {epoch_time:.1f}s | ETA: {remaining/60:.1f}min')
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history, time_elapsed

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, dataloader, class_names, device, results_dir, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with autocast('cuda', enabled=(device.type == "cuda")):
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(cm, class_names, os.path.join(results_dir, f"{model_name}_cm.png"), model_name)
    
    return acc

def save_confusion_matrix(cm, class_names, path, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_history(history, model_name, results_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['val_acc'], label='Val')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.title(f'{model_name} Loss')
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, f"{model_name}_history.png"))
    plt.close()

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_features(dataloader, device):
    print("  Extracting features with EfficientNetB0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        with autocast('cuda', enabled=(device.type == "cuda")):
            for inputs, y in tqdm(dataloader, desc="  Extracting features"):
                inputs = inputs.to(device, non_blocking=True)
                feats = model(inputs)
                features.append(feats.float().cpu().numpy())
                labels.append(y.numpy())
            
    return np.vstack(features), np.concatenate(labels)

def extract_features_cached(dataloader, device, cache_path, name=""):
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]
    
    print(f"  Extracting {name} features (will cache to {cache_path})...")
    X, y = extract_features(dataloader, device)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("GARBAGE CLASSIFICATION - MODEL COMPARISON")
    print("="*60)
    
    set_seeds(SEED)
    setup_cuda_optimizations()
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\n[INFO] Loading data...")
    dataloaders, feature_loaders, class_names = prepare_datasets()
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")
    print(f"[INFO] Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    
    results = []
    
    # -------------------------------------------------------------------------
    # PHASE 1: Feature Extraction (kNN, MLP) - using EVAL transforms!
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: Feature Extraction (eval transforms, no augmentation)")
    print("="*60)
    
    # Delete old feature caches (they used wrong transforms)
    train_cache = os.path.join(RESULTS_DIR, "train_feats.npz")
    val_cache = os.path.join(RESULTS_DIR, "val_feats.npz")
    test_cache = os.path.join(RESULTS_DIR, "test_feats.npz")
    for cache in [train_cache, val_cache, test_cache]:
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  [INFO] Deleted old cache: {cache}")
    
    # Extract features using EVAL transforms (consistent with Flask inference)
    train_feats, train_labels = extract_features_cached(feature_loaders['train'], device, train_cache, "train")
    val_feats, val_labels = extract_features_cached(feature_loaders['val'], device, val_cache, "val")
    test_feats, test_labels = extract_features_cached(feature_loaders['test'], device, test_cache, "test")
    
    # kNN
    print("\n[Training kNN]")
    knn_start = time.time()
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    test_feats_scaled = scaler.transform(test_feats)
    
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_feats_scaled, train_labels)
    knn_preds = knn.predict(test_feats_scaled)
    knn_acc = accuracy_score(test_labels, knn_preds)
    knn_time = time.time() - knn_start
    print(f"  kNN Test Accuracy: {knn_acc:.4f} ({knn_time:.1f}s)")
    results.append({'model': 'kNN', 'test_acc': knn_acc, 'time': knn_time})
    
    # MLP
    print("\n[Training MLP]")
    mlp = MLP(train_feats.shape[1], num_classes).to(device)
    mlp_optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    feat_train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(train_feats), torch.LongTensor(train_labels))
    feat_val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(val_feats), torch.LongTensor(val_labels))
    feat_train_loader = DataLoader(feat_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    feat_val_loader = DataLoader(feat_val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    mlp, mlp_hist, mlp_time = train_model(
        mlp, {'train': feat_train_loader, 'val': feat_val_loader},
        criterion, mlp_optimizer, num_epochs=EPOCHS, device=device, model_name="MLP"
    )
    
    mlp.eval()
    with torch.no_grad():
        mlp_preds = mlp(torch.FloatTensor(test_feats).to(device))
        mlp_pred_cls = torch.argmax(mlp_preds, dim=1).cpu().numpy()
        mlp_acc = accuracy_score(test_labels, mlp_pred_cls)
    
    print(f"  MLP Test Accuracy: {mlp_acc:.4f}")
    results.append({'model': 'MLP', 'test_acc': mlp_acc, 'time': mlp_time})
    plot_history(mlp_hist, "MLP", RESULTS_DIR)
    torch.save(mlp.state_dict(), os.path.join(RESULTS_DIR, "mlp_classifier.pth"))
    print(f"  Saved: mlp_classifier.pth")

    # -------------------------------------------------------------------------
    # PHASE 2: CNN Models
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: End-to-End CNN Training")
    print("="*60)
    
    for model_name, save_name in [("simple_cnn", "simple_cnn.pth"), ("mobilenet", "mobilenet.pth")]:
        print(f"\n[Training {model_name}]")
        
        model = build_model(model_name, num_classes, device)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model, history, duration = train_model(
            model, dataloaders, criterion, optimizer,
            num_epochs=EPOCHS, device=device, model_name=model_name
        )
        
        acc = evaluate_model(model, dataloaders['test'], class_names, device, RESULTS_DIR, model_name)
        print(f"  {model_name} Test Accuracy: {acc:.4f}")
        results.append({'model': model_name, 'test_acc': acc, 'time': duration})
        plot_history(history, model_name, RESULTS_DIR)
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, save_name))
        print(f"  Saved: {save_name}")
    
    # EfficientNet (frozen backbone only)
    print("\n[Training EfficientNet (frozen backbone)]")
    effnet = build_model("efficientnet", num_classes, device)
    trainable_params = filter(lambda p: p.requires_grad, effnet.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    effnet, effnet_hist, effnet_time = train_model(
        effnet, dataloaders, criterion, optimizer,
        num_epochs=EPOCHS, device=device, model_name="efficientnet"
    )
    
    effnet_acc = evaluate_model(effnet, dataloaders['test'], class_names, device, RESULTS_DIR, "efficientnet")
    print(f"  EfficientNet Test Accuracy: {effnet_acc:.4f}")
    results.append({'model': 'EfficientNet', 'test_acc': effnet_acc, 'time': effnet_time})
    plot_history(effnet_hist, "efficientnet", RESULTS_DIR)
    torch.save(effnet.state_dict(), os.path.join(RESULTS_DIR, "efficientnet_best.pth"))
    print(f"  Saved: efficientnet_best.pth")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<20} | {'Test Acc':<10} | {'Time (s)':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['model']:<20} | {r['test_acc']:.4f}     | {r['time']:.1f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot comparison
    fig, ax1 = plt.subplots(figsize=(12, 6))
    names = [r['model'] for r in results]
    accs = [r['test_acc'] for r in results]
    times = [r['time'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, accs, width, label='Accuracy', color='C0')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.0)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, times, width, label='Time (s)', color='C1')
    ax2.set_ylabel('Time (s)')
    
    plt.title("Model Comparison (RTX 4060 Optimized)")
    plt.xticks(x, names, rotation=15)
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_summary.png"))
    
    print(f"\n[INFO] Results saved to {RESULTS_DIR}")
    print("[INFO] Training complete!")

if __name__ == "__main__":
    main()
