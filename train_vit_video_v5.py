import os, math, json, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from timm.utils.model_ema import ModelEmaV2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.amp import autocast, GradScaler  # new API

# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# NumPy-only image scoring 
# ----------------------------
_SOBEL_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
_LAPLACE = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

def _conv2d(img2d: np.ndarray, k: np.ndarray) -> np.ndarray:
    H, W = img2d.shape
    kh, kw = k.shape
    ph, pw = kh//2, kw//2
    pad = np.pad(img2d, ((ph,ph),(pw,pw)), mode="edge")
    out = np.empty_like(img2d, dtype=np.float32)
    # simple reference 2D conv (fast enough for 8 frames / sample)
    for i in range(H):
        ii = i
        for j in range(W):
            patch = pad[ii:ii+kh, j:j+kw]
            out[i, j] = float((patch * k).sum())
    return out

def frame_score_pil(img: Image.Image) -> float:
    """Higher=more informative frame (sharp, textured, edged)."""
    x = np.asarray(img.convert("L"), dtype=np.float32)  # [H,W]
    gx = _conv2d(x, _SOBEL_X)
    gy = _conv2d(x, _SOBEL_Y)
    grad = np.hypot(gx, gy)

    lap = _conv2d(x, _LAPLACE)
    sharp = float(lap.var() + grad.var())

    thr = np.percentile(grad, 90.0)
    edge_density = float((grad >= thr).mean())  # 0..1
    std = float(x.std())

    return 0.6 * sharp + 0.3 * std + 0.1 * (edge_density * 100.0)

def is_empty_pil(img: Image.Image, thr_std=2.0, frac=0.985) -> bool:
    x = np.asarray(img.convert("L"), dtype=np.float32)
    if x.std() < thr_std:
        return True
    p16, p84 = np.percentile(x, 16), np.percentile(x, 84)
    if (p84 - p16) < 3:
        return True
    if (x <= 4).mean() > frac or (x >= 251).mean() > frac:
        return True
    return False

# ----------------------------
# Dataset: video-level (8 frames per sample)
# CSV columns: frame_path, video_id, pmos
# ----------------------------
class VideoDataset(Dataset):
    def __init__(self, csv_path, img_size=224, is_train=True, frames_per_video=8):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        self.T = frames_per_video
        self._vids = self.df["video_id"].unique()

        base = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ]
        aug = []
        if is_train:
            aug = [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1,0.1,0.1,0.05),
                transforms.RandomGrayscale(p=0.1),
            ]
        self.tfms = transforms.Compose(aug + base)

    def __len__(self): 
        return len(self._vids)

    def _rows_for_video(self, vid):
        rows = self.df[self.df["video_id"] == vid].copy()
        # stable order helps sampling
        rows = rows.sort_values("frame_path")
        return rows

    def __getitem__(self, idx):
        vid = self._vids[idx]
        rows = self._rows_for_video(vid)
        target = float(rows["pmos"].iloc[0])

        # score frames & drop empties
        scored = []
        for p in rows["frame_path"].tolist():
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            if is_empty_pil(im):
                continue
            scored.append((frame_score_pil(im), im))

        if len(scored) == 0:
            # fallback: if all were empty/unreadable, just load what we can
            for p in rows["frame_path"].tolist():
                try:
                    im = Image.open(p).convert("RGB")
                    scored.append((frame_score_pil(im), im))
                except Exception:
                    pass

        # top → evenly spaced T frames
        scored.sort(key=lambda t: t[0], reverse=True)
        if len(scored) >= self.T:
            idxs = np.linspace(0, len(scored)-1, self.T, dtype=int)
            imgs   = [scored[i][1] for i in idxs]
            scores = [scored[i][0] for i in idxs]
        else:
            imgs   = [s[1] for s in scored]
            scores = [s[0] for s in scored]
            while len(imgs) < self.T:
                imgs.append(imgs[-1])
                scores.append(scores[-1])

        tens = [self.tfms(im) for im in imgs]     # T * [3,H,W]
        x = torch.stack(tens, dim=0)              # [T,3,H,W]

        sc = np.asarray(scores, dtype=np.float32)
        sc = (sc - sc.mean()) / (sc.std() + 1e-6)
        sc = torch.tensor(sc, dtype=torch.float32)  # [T]

        y = torch.tensor([target], dtype=torch.float32)
        return x, y, str(vid), sc

# ----------------------------
# Model: ViT backbone + attention pooling across frames
# ----------------------------
class ViTVideoRegressor(nn.Module):
    def __init__(self, arch="vit_small_patch16_224", num_classes=1, drop_path=0.15):
        super().__init__()
        self.backbone = timm.create_model(
            arch, pretrained=True, num_classes=0, drop_path_rate=drop_path
        )
        hdim = self.backbone.num_features
        self.att = nn.Sequential(
            nn.LayerNorm(hdim),
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Linear(hdim // 2, 1)
        )
        self.head = nn.Linear(hdim, num_classes)

    def _feat_pool(self, feats):
        """
        timm.forward_features can return:
        - [N, D] (already pooled), or
        - [N, tokens, D] (per-token). We pool to [N, D].
        - sometimes tuples/dicts; we extract the main tensor.
        """
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        elif isinstance(feats, dict):
            # common keys: 'x', 'features', 'out'
            for k in ("x", "features", "out"):
                if k in feats and torch.is_tensor(feats[k]):
                    feats = feats[k]; break

        if feats.dim() == 3:
            # Use CLS token; alternative: feats[:,1:,:].mean(1)
            feats = feats[:, 0, :]
        return feats  # [N, D]

    def forward(self, x, score_prior=None):
        # x: [B,T,3,H,W]
        B, T = x.shape[0], x.shape[1]
        x = x.view(B*T, x.size(2), x.size(3), x.size(4))
        feats = self.backbone.forward_features(x)     # possibly [B*T, tokens, D]
        feats = self._feat_pool(feats)                # [B*T, D]
        D = feats.shape[-1]
        feats = feats.view(B, T, D)                   # [B,T,D]

        att_logits = self.att(feats).squeeze(-1)      # [B,T]
        if score_prior is not None:                   # nudge attention with our score
            att_logits = att_logits + 0.1 * score_prior
        w = torch.softmax(att_logits, dim=1)          # [B,T]
        pooled = torch.einsum("btd,bt->bd", feats, w) # [B,D]
        y = self.head(pooled).squeeze(-1)             # [B]
        return y

# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, vids = [], [], []
    for x, y, v, sc in loader:
        x = x.to(device)
        sc = sc.to(device)
        out = model(x, score_prior=sc).float().cpu().numpy()
        y_pred += out.tolist()
        y_true += y.squeeze(1).cpu().numpy().tolist()
        vids   += list(v)

    df = pd.DataFrame({"video_id": vids, "y_true": y_true, "y_pred": y_pred})
    mae  = float(mean_absolute_error(df["y_true"], df["y_pred"]))
    rmse = float(math.sqrt(mean_squared_error(df["y_true"], df["y_pred"])))
    plcc = float(np.corrcoef(df["y_true"], df["y_pred"])[0, 1])
    srcc = float(pd.Series(df["y_true"]).rank().corr(
        pd.Series(df["y_pred"]).rank(), method="spearman"))
    ss_res = float(((df["y_true"]-df["y_pred"])**2).sum())
    ss_tot = float(((df["y_true"]-df["y_true"].mean())**2).sum())
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return {"video_mae": mae, "video_rmse": rmse, "video_plcc": plcc, "video_srcc": srcc, "video_r2": r2}

# ----------------------------
# Train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=12)   # videos per batch
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--drop_path", type=float, default=0.15)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr = VideoDataset(args.train_csv, args.img_size, is_train=True,  frames_per_video=8)
    ds_vl = VideoDataset(args.val_csv,   args.img_size, is_train=False, frames_per_video=8)
    ds_te = VideoDataset(args.test_csv,  args.img_size, is_train=False, frames_per_video=8)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_vl = DataLoader(ds_vl, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = ViTVideoRegressor(args.arch, num_classes=1, drop_path=args.drop_path).to(device)
    ema = ModelEmaV2(model, decay=0.9998)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - 1))
    sched  = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[1])

    loss_fn = nn.SmoothL1Loss(beta=0.5)
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    best_rmse = 1e9

    from tqdm import tqdm
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, v, sc in tqdm(dl_tr, desc=f"Epoch {ep}/{args.epochs}"):
            x  = x.to(device)
            y  = y.squeeze(1).to(device)
            sc = sc.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(device.type == "cuda")):
                pred = model(x, score_prior=sc)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            losses.append(float(loss.item()))

        # validation with EMA weights
        metrics = evaluate(ema.module, dl_vl, device)
        with open(out_dir / "val_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": ep, **metrics}) + "\n")

        torch.save(model.state_dict(), str(out_dir / "last.pt"))
        if metrics["video_rmse"] < best_rmse:
            best_rmse = metrics["video_rmse"]
            torch.save(ema.module.state_dict(), str(out_dir / "best_model.pt"))

        print(f"Epoch {ep}/{args.epochs}  loss={np.mean(losses):.4f}  "
              f"[Val] vMAE={metrics['video_mae']:.4f}  vRMSE={metrics['video_rmse']:.4f}  "
              f"PLCC={metrics['video_plcc']:.4f}  SRCC={metrics['video_srcc']:.4f}  R2={metrics['video_r2']:.4f}")

        sched.step()

    # final test with EMA best
    ema.module.load_state_dict(torch.load(str(out_dir / "best_model.pt"), map_location=device))
    test_metrics = evaluate(ema.module, dl_te, device)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print("[Test]", test_metrics)

if __name__ == "__main__":
    main()
