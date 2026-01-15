# client_eval_v5_tta_cal.py
# Re-evaluate v5 model with TTA (orig + H-flip + temporal-reverse)
# + post-hoc linear calibration (fit on VAL, apply to TEST).
# Matches train_vit_video_v5.py architecture (temporal_score + 5-layer head).

import os, json, math, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error

import timm


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset (CSV columns: frame_path, video_id, pmos)
# Deterministic, eval-only transforms. Frames picked evenly.
# ----------------------------
class VideoDataset(Dataset):
    def __init__(self, csv_path, img_size=224, is_train=False, frames_per_video=8):
        self.df = pd.read_csv(csv_path)
        self.T = frames_per_video
        self._vids = self.df["video_id"].astype(str).unique()

        self.tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])

    def __len__(self):
        return len(self._vids)

    def _rows_for_video(self, vid):
        rows = self.df[self.df["video_id"].astype(str) == vid].copy()
        rows = rows.sort_values("frame_path")
        return rows

    def _pick_T_evenly(self, items):
        # items: list of PIL images
        if len(items) >= self.T:
            idxs = np.linspace(0, len(items)-1, self.T, dtype=int)
            return [items[i] for i in idxs]
        out = items[:]
        while len(out) < self.T:
            out.append(out[-1])
        return out

    def __getitem__(self, idx):
        vid = self._vids[idx]
        rows = self._rows_for_video(vid)
        target = float(rows["pmos"].iloc[0])

        imgs = []
        for p in rows["frame_path"].tolist():
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(im)
            except Exception:
                pass

        if len(imgs) == 0:
            # fallback gray dummy to avoid crash (should be rare)
            imgs = [Image.fromarray(np.full((224,224,3), 128, np.uint8))]

        imgs = self._pick_T_evenly(imgs)
        x = torch.stack([self.tfms(im) for im in imgs], dim=0)  # [T,3,H,W]
        y = torch.tensor([target], dtype=torch.float32)
        return x, y, vid


# ----------------------------
# Model (exactly match train_vit_video_v5.py)
# ViT features + temporal_score (LN->Linear) + 5-layer head
# ----------------------------
class ViTVideoRegressor(nn.Module):
    def __init__(self, arch="vit_small_patch16_224", drop_path=0.15):
        super().__init__()
        # Feature extractor (no classifier head)
        self.backbone = timm.create_model(
            arch, pretrained=False, num_classes=0, drop_path_rate=drop_path
        )
        hdim = self.backbone.num_features

        # SAME MODULE & PARAM NAMES as training checkpoint
        self.temporal_score = nn.Sequential(
            nn.LayerNorm(hdim),
            nn.Linear(hdim, 1)      # logits per frame
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hdim),
            nn.Dropout(0.1),
            nn.Linear(hdim, hdim // 2),
            nn.GELU(),
            nn.Linear(hdim // 2, 1),  # scalar MOS
        )

    def _feat_pool(self, feats):
        # timm.forward_features may return [N,D], [N,Tk,D], or dict/tuple
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        elif isinstance(feats, dict):
            for k in ("x", "features", "out"):
                if k in feats and torch.is_tensor(feats[k]):
                    feats = feats[k]
                    break
        if feats.dim() == 3:
            feats = feats[:, 0, :]  # CLS token
        return feats  # [N, D]

    def forward(self, x):  # x: [B,T,3,H,W]
        B, T = x.shape[:2]
        x = x.view(B*T, x.size(2), x.size(3), x.size(4))
        feats = self.backbone.forward_features(x)
        feats = self._feat_pool(feats)        # [B*T, D]
        D = feats.shape[-1]
        feats = feats.view(B, T, D)           # [B, T, D]

        logits = self.temporal_score(feats).squeeze(-1)  # [B, T]
        w = torch.softmax(logits, dim=1)                 # [B, T]
        pooled = torch.einsum("btd,bt->bd", feats, w)    # [B, D]
        y = self.head(pooled).squeeze(1)                 # [B]
        return y


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    plcc = float(np.corrcoef(y_true, y_pred)[0,1]) if len(y_true) > 1 else float("nan")
    srcc = pd.Series(y_true).rank().corr(pd.Series(y_pred).rank(), method="spearman")
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    acc05 = float(np.mean(np.abs(y_pred - y_true) <= 0.05))
    acc10 = float(np.mean(np.abs(y_pred - y_true) <= 0.10))
    return {
        "video_mae": float(mae),
        "video_rmse": float(rmse),
        "video_plcc": float(plcc),
        "video_srcc": float(srcc),
        "video_r2": float(r2),
        "acc@0.05": acc05,
        "acc@0.10": acc10,
    }


# ----------------------------
# TTA prediction (orig + H-flip + temporal-reverse)
# ----------------------------
@torch.no_grad()
def predict_with_tta(model, loader, device):
    model.eval()
    yt, yp, vids = [], [], []
    for xb, yb, vb in loader:
        xb = xb.to(device)  # [B,T,3,H,W]

        p0 = model(xb)
        pH = model(torch.flip(xb, dims=[4]))  # horizontal flip (flip W-dim)
        pT = model(torch.flip(xb, dims=[1]))  # temporal reverse (flip T-dim)

        pred = (p0 + pH + pT) / 3.0
        yp.extend(pred.float().cpu().numpy().tolist())
        yt.extend(yb.squeeze(1).cpu().numpy().tolist())
        vids.extend(vb)
    return vids, yt, yp


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--weights",   required=True)  # path to best_model.pt
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--drop_path", type=float, default=0.15)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    ds_val  = VideoDataset(args.val_csv,  args.img_size, is_train=False, frames_per_video=args.frames_per_video)
    ds_test = VideoDataset(args.test_csv, args.img_size, is_train=False, frames_per_video=args.frames_per_video)
    dl_val  = DataLoader(ds_val,  batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Model (match training)
    model = ViTVideoRegressor(args.arch, drop_path=args.drop_path).to(device)

    # Load EMA/best weights you trained
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)

    print("Device:", device.type)
    print(f"Loaded weights from: {args.weights}")

    # VAL (uncalibrated) → calibration fit
    v_vids, v_true, v_pred = predict_with_tta(model, dl_val, device)
    val_uncal = compute_metrics(v_true, v_pred)
    X = np.vstack([np.asarray(v_pred), np.ones(len(v_pred))]).T
    y = np.asarray(v_true)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"Calibration (VAL): y_true ≈ {a:.4f} * y_pred + {b:.4f}")

    # TEST (uncalibrated)
    t_vids, t_true, t_pred = predict_with_tta(model, dl_test, device)
    test_uncal = compute_metrics(t_true, t_pred)

    # TEST (calibrated)
    t_pred_cal = (a * np.asarray(t_pred) + b).tolist()
    test_cal = compute_metrics(t_true, t_pred_cal)

    # Save report + predictions
    report = {
        "VAL (uncal)": val_uncal,
        "calibration": {"a": float(a), "b": float(b)},
        "TEST (uncal)": test_uncal,
        "TEST (cal)": test_cal,
        "N_val": len(v_true),
        "N_test": len(t_true),
    }
    with open(out_dir / "client_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    pd.DataFrame({
        "video_id": t_vids,
        "y_true": t_true,
        "y_pred_uncal": t_pred,
        "y_pred_cal": t_pred_cal
    }).to_csv(out_dir / "client_eval_preds_test.csv", index=False)


if __name__ == "__main__":
    main()
