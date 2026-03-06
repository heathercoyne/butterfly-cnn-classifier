# version 2.2
import argparse, os, random, json, csv
from pathlib import Path
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix


# for reproducibility seed set to 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# adding transformations to data (version 2 has some extra augmentation)
def make_transforms(img_size=64, strong_aug=False):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # can have stronger jitter if chosen when i run 
    if strong_aug:
        jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.08)
        scale  = (0.5, 1.0)
    else:
        jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        scale  = (0.6, 1.0)
        
    #random transformations for training data
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomHorizontalFlip(p=0.5),
        jitter,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    #set transformations for evaluation
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# randomly splitting the data into parts fo training (70%), val (15%), and test (15%)
def random_indices_by_ratio(n_samples, ratios=(0.70, 0.15, 0.15), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idxs = np.arange(n_samples)
    rng.shuffle(idxs)
    n = len(idxs)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    tr = idxs[:n_tr].tolist()
    va = idxs[n_tr:n_tr+n_va].tolist()
    te = idxs[n_tr+n_va:].tolist()
    return tr, va, te


def build_datasets_autosplit(data_dir, img_size=64, seed=42, strong_aug=False):
    train_tf, eval_tf = make_transforms(img_size, strong_aug=strong_aug)
    base_plain = ImageFolder(data_dir)
    classes = base_plain.classes

    tr_idx, va_idx, te_idx = random_indices_by_ratio(
        len(base_plain.samples), ratios=(0.70, 0.15, 0.15), seed=seed
    )
    ds_train_full = ImageFolder(data_dir, transform=train_tf)
    ds_eval_full  = ImageFolder(data_dir, transform=eval_tf)

    return (Subset(ds_train_full, tr_idx),
            Subset(ds_eval_full, va_idx),
            Subset(ds_eval_full, te_idx),
            classes)


#  CNN with batch normailization and then ReLU 
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


# with old downsampling CNN, batch normalization, and dropout 
# in this version a pooling layer was added 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int, channels=(32, 64, 96, 128, 128), p_drop=0.5,
                 use_pool=False, pool_type="max"):  
        super().__init__()
        c1, c2, c3, c4, c5 = channels

        # helper to create a pooling layer
        def pool():
            if pool_type == "avg":
                return nn.AvgPool2d(kernel_size=2, stride=2)
            return nn.MaxPool2d(kernel_size=2, stride=2)

        if use_pool:
            # When using pooling, keep conv stride=1, then downsample by 2x pooling.
            # 64 -> 32
            b1 = nn.Sequential(conv_bn_relu(3,  c1, s=1), pool())
            # 32 -> 16
            b2 = nn.Sequential(conv_bn_relu(c1, c2, s=1), pool())
            # 16 -> 8
            b3 = nn.Sequential(conv_bn_relu(c2, c3, s=1), pool())
            # 8 -> 4
            b4 = nn.Sequential(conv_bn_relu(c3, c4, s=1), pool())
        else:
            # if no pooling (ill compare when i run): stride-2 convs do the downsampling
            # 64 -> 32
            b1 = conv_bn_relu(3,  c1, s=2)   # stride-2 downsample
            # 32 -> 16
            b2 = conv_bn_relu(c1, c2, s=2)
            # 16 -> 8
            b3 = conv_bn_relu(c2, c3, s=2)
            # 8 -> 4
            b4 = conv_bn_relu(c3, c4, s=2)

        # refine at 4x4 
        b5 = conv_bn_relu(c4, c5, s=1)

        self.features = nn.Sequential(b1, b2, b3, b4, b5)

        # compute final spatial size
        assert img_size % 16 == 0, "img_size should be divisible by 16"
        final_sp = img_size // 16  
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c5 * final_sp * final_sp, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Evaluation, calculating loss and accuracy
@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    tot_loss, tot_n, tot_correct = 0.0, 0, 0
    y_true_all, y_pred_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(1)
        tot_loss += loss.item() * y.size(0)
        tot_correct += (preds == y).sum().item()
        tot_n += y.size(0)
        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=int)
    return {"loss": tot_loss / max(1, tot_n),
            "acc":  tot_correct / max(1, tot_n),
            "y_true": y_true, "y_pred": y_pred}


#  Confusion matrix plot (for the final report)
def save_confusion_matrix(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    thr = cm.max()/2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


#  Per-class accuracy 
def per_class_accuracy(y_true, y_pred, num_classes):
    rows = []
    for c in range(num_classes):
        mask = (y_true == c)
        support = int(mask.sum())
        correct = int(((y_pred == c) & mask).sum())
        acc = (correct / support) if support > 0 else None
        rows.append({"class_id": c, "support": support, "correct": correct, "accuracy": acc})
    return rows


def save_per_class_csv(rows, class_names, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "support", "correct", "accuracy"])
        for r in rows:
            cname = class_names[r["class_id"]] if 0 <= r["class_id"] < len(class_names) else str(r["class_id"])
            acc_str = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A"
            w.writerow([r["class_id"], cname, r["support"], r["correct"], acc_str])


# Training and writing out results (all to folder output)
def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)

    tr_ds, va_ds, te_ds, classes = build_datasets_autosplit(
        args.data_dir, img_size=args.img_size, seed=42, strong_aug=args.strong_aug
    )
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    channels = (32, 64, 96, 128, 128)

    model = SimpleCNN(num_classes=len(classes), img_size=args.img_size,
                      channels=channels, p_drop=args.dropout,
                      use_pool=args.use_pool, pool_type=args.pool_type).to(device)  
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": 42, "device": str(device),
            "split": "random 70/15/15",
            "arch": f"CNN (train2.2)",
            "channels": channels,
            "params": count_params(model),
            "optimizer": "sgd",
            "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "img_size": args.img_size, "classes": classes,
            "strong_aug": bool(args.strong_aug), "dropout": args.dropout,
            "use_pool": bool(args.use_pool), "pool_type": args.pool_type   
        }, f, indent=2)

    best_val_acc = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        seen, run_loss, run_acc_sum = 0, 0.0, 0.0

        pbar = tqdm(tr, desc=f"Epoch {ep}/{args.epochs}", dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc_b = (logits.argmax(1) == y).float().mean().item()
            b = y.size(0)
            seen += b; run_loss += loss.item() * b; run_acc_sum += acc_b * b
            pbar.set_postfix(loss=f"{run_loss/seen:.4f}", acc=f"{run_acc_sum/seen:.4f}")

        val_stats = evaluate(model, va, device, loss_fn)
        print(f"Epoch {ep}/{args.epochs} | "
              f"train_loss={run_loss/seen:.4f} train_acc={run_acc_sum/seen:.4f} | "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}", flush=True)

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save({"model": model.state_dict(), "classes": classes, "channels": channels}, out / "best.pt")
            print("Saved best model (by val accuracy).", flush=True)

    # Test results to ./output 
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_stats = evaluate(model, te, device, loss_fn)

    # metrics.txt
    with open(out / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("CNN (train2.2) - Test Metrics)\n")
        f.write(f"Params: {count_params(model)}\n")
        f.write(f"Loss: {test_stats['loss']:.4f}\n")
        f.write(f"Accuracy: {test_stats['acc']:.4f}\n")

    # per_class_metrics.csv 
    per_cls = per_class_accuracy(test_stats["y_true"], test_stats["y_pred"], num_classes=len(classes))
    with open(out / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["class_id", "class_name", "support", "correct", "accuracy"])
        for i, row in enumerate(per_cls):
            cname = classes[row["class_id"]]
            acc_str = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "N/A"
            w.writerow([row["class_id"], cname, row["support"], row["correct"], acc_str])
    with open(out / "metrics.txt", "a", encoding="utf-8") as f:
        f.write("\nPer-class accuracy (class_name: acc [correct/support]):\n")
        for row in per_cls:
            cname = classes[row["class_id"]]
            if row["accuracy"] is None:
                f.write(f"- {cname}: N/A (no test samples)\n")
            else:
                f.write(f"- {cname}: {row['accuracy']:.4f}  [{row['correct']}/{row['support']}]\n")

    # confusion_matrix.png
    cm = confusion_matrix(test_stats["y_true"], test_stats["y_pred"], labels=list(range(len(classes))))
    save_confusion_matrix(cm, classes, out / "confusion_matrix.png")

    print(f"[TEST] loss={test_stats['loss']:.4f} acc={test_stats['acc']:.4f}")
    print(f"Saved artifacts in: {out.resolve()}")


# for command line (i changed defaults based on the best parameters i got when i ran version 2.1)
def parse_args():
    p = argparse.ArgumentParser(description="CNN (train2.2)")
    p.add_argument("--data_dir", type=str, default="./ButterflyClassificationDataset")
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size",   type=int, default=128)    # must be divisible by 16
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--momentum",   type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--strong_aug", action="store_true")
    p.add_argument("--use_pool", action="store_true")             
    p.add_argument("--pool_type", type=str, default="max", choices=["max", "avg"])  
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

