"""Train, select, test and export one model.

    uv run python -m impactlens.models.train --task oscd --data build/kaggle --out runs/oscd
    uv run python -m impactlens.models.train --task floods --data build/kaggle --out runs/floods

What happens, in order:
 1. Train. After every epoch score the VALIDATION split (loss, IoU, F1, precision, recall).
    The best epoch (by validation F1) is saved as best.pt; training stops early when validation
    F1 has not improved for ``--patience`` epochs.
 2. Load best.pt and score the TEST split once (floods: also the Bolivia split). Test data is
    not touched before this point, so it can never influence checkpoint choice.
 3. floods only: score the NDWI-threshold baseline on the same splits.
 4. Export best.pt to ONNX and check ONNX output == PyTorch output.
Everything goes to <out>/metrics.json; validation and test numbers are stored separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from impactlens.data.oscd import OSCDDataset
from impactlens.data.sen1floods11 import Sen1Floods11Dataset
from impactlens.models import baseline, export
from impactlens.models.losses import bce_dice_loss
from impactlens.models.metrics import ConfusionMeter
from impactlens.models.siamese_unet import SiameseUNetDiff
from impactlens.models.unet import UNet

TASKS = ("oscd", "floods")
VAL_SPLIT = {"oscd": "val", "floods": "valid"}
FINAL_SPLITS = {"oscd": ("test",), "floods": ("test", "bolivia")}  # scored once, after training
DEFAULT_BASE = {"oscd": 16, "floods": 32}
NPZ_NAME = {"oscd": "oscd.npz", "floods": "sen1floods11.npz"}


# ------------------------------------------------------------------------------ plumbing


def pick_device(name: str = "auto") -> torch.device:
    """cuda (Kaggle) > mps (Apple Silicon) > cpu, unless a device is forced."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_data(task: str, data: str | Path) -> Path:
    """``data`` may be a packed .npz, a folder holding it, or the raw dataset folder."""
    p = Path(data)
    if p.suffix != ".npz" and (p / NPZ_NAME[task]).exists():
        return p / NPZ_NAME[task]
    return p


def build_model(task: str, base: int | None = None) -> torch.nn.Module:
    base = base or DEFAULT_BASE[task]
    return SiameseUNetDiff(base=base) if task == "oscd" else UNet(base=base)


def make_dataset(task, source, split, tile, train=False, crops=None):
    if task == "oscd":
        return OSCDDataset(
            source, split, tile=tile, augment=train, crops_per_epoch=crops if train else None
        )
    return Sen1Floods11Dataset(source, split, tile=tile, augment=train)


def forward(task: str, model: torch.nn.Module, batch: dict, device) -> torch.Tensor:
    if task == "oscd":
        return model(batch["img1"].to(device), batch["img2"].to(device))
    return model(batch["img"].to(device))


@torch.no_grad()
def evaluate(task, model, loader, device, amp=False, max_batches=None) -> dict[str, float]:
    """Loss + IoU/F1/precision/recall over a whole split (fixed tiles, valid pixels only)."""
    model.eval()
    meter, total, n = ConfusionMeter(), 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        mask, valid = batch["mask"].to(device), batch["valid"].to(device)
        with torch.autocast(device.type, enabled=amp):
            logits = forward(task, model, batch, device)
        total += float(bce_dice_loss(logits, mask, valid))
        n += 1
        meter.update(logits.float(), mask, valid)
    return {"loss": total / max(n, 1), **meter.compute()}


# ------------------------------------------------------------------------------ main flow


def train(args) -> dict:
    task, out = args.task, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp  # mixed precision only on NVIDIA GPUs
    source = resolve_data(task, args.data)
    cap = args.max_batches  # only set for smoke runs

    train_ds = make_dataset(task, source, "train", args.tile, train=True, crops=args.crops)
    val_ds = make_dataset(task, source, VAL_SPLIT[task], args.tile)
    kw = {"batch_size": args.batch_size, "num_workers": args.workers}
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **kw)
    val_dl = DataLoader(val_ds, **kw)

    model = build_model(task, args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print(f"task={task} device={device} amp={amp} params={n_params / 1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)} tiles")  # fmt: skip

    history, best, best_epoch = [], None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i, batch in enumerate(train_dl):
            if cap is not None and i >= cap:
                break
            mask, valid = batch["mask"].to(device), batch["valid"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = forward(task, model, batch, device)
            loss = bce_dice_loss(logits, mask, valid)  # computed in float32
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        val = evaluate(task, model, val_dl, device, amp, cap)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val,
               "seconds": round(time.time() - t0, 1)}  # fmt: skip
        history.append(row)
        print(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} | val loss {val['loss']:.4f} "
              f"IoU {val['iou']:.3f} F1 {val['f1']:.3f} P {val['precision']:.3f} "
              f"R {val['recall']:.3f} | {row['seconds']}s")  # fmt: skip
        if best is None or val["f1"] > best["f1"]:  # checkpoint chosen on validation F1 only
            best, best_epoch = val, epoch
            torch.save({"task": task, "base": args.base or DEFAULT_BASE[task],
                        "epoch": epoch, "val": val, "state_dict": model.state_dict()},
                       out / "best.pt")  # fmt: skip
        elif epoch - best_epoch >= args.patience:
            print(f"early stop: no validation F1 gain for {args.patience} epochs")
            break

    # ---- final, one-time evaluation of the best checkpoint on the held-out splits ----
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(task, ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    final = {}
    for split in FINAL_SPLITS[task]:
        ds = make_dataset(task, source, split, args.tile)
        final[split] = evaluate(task, model, DataLoader(ds, **kw), device, amp, cap)
        m = final[split]
        print(f"FINAL {split:8} | IoU {m['iou']:.3f} F1 {m['f1']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f}")  # fmt: skip

    metrics = {
        "task": task,
        "device": str(device),
        "amp": amp,
        "params": n_params,
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "history": history,
        "best_epoch": best_epoch,
        "val": history[best_epoch - 1]["val"],  # validation: what checkpoint selection saw
        **final,  # "test" (and "bolivia"): scored once, kept apart from val
    }
    if task == "floods":
        others = {s: make_dataset(task, source, s, args.tile) for s in FINAL_SPLITS[task]}
        metrics["baseline_ndwi"] = baseline.run_baseline(val_ds, others, cap)
        b = metrics["baseline_ndwi"]
        for s in others:
            f, t = b[s]["fixed"]["f1"], b[s]["tuned"]["f1"]
            print(f"NDWI baseline {s:8} | F1 {f:.3f} (threshold 0)  {t:.3f} (tuned on val)")

    # ---- ONNX export + parity check, using a real validation batch ----
    batch = next(iter(DataLoader(val_ds, batch_size=min(4, len(val_ds)))))
    inputs = [batch[k] for k in export.input_names(task)]
    onnx_path = export.export_onnx(model, task, out / f"{task}.onnx")
    metrics["onnx"] = export.verify_onnx(model, task, onnx_path, inputs)
    o = metrics["onnx"]
    verdict = "OK" if o["ok"] else "MISMATCH"
    print(
        f"ONNX check: max logit diff {o['max_abs_logit_diff']:.2e} "
        f"(limit {o['tolerance']}) {verdict}"
    )

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out}/best.pt, {onnx_path.name}, metrics.json")
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--data", required=True, help="packed .npz, folder holding it, or raw folder")
    p.add_argument(
        "--out", required=True, help="output folder (best.pt, <task>.onnx, metrics.json)"
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument(
        "--patience", type=int, default=10, help="early-stop after N epochs w/o val gain"
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--base", type=int, default=None, help="channel width (default 16 oscd, 32 floods)"
    )
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--crops", type=int, default=400, help="oscd: random crops per epoch")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="SMOKE TESTS ONLY: cap batches")
    return p.parse_args(argv)


def main(argv=None) -> None:
    metrics = train(parse_args(argv))
    if not metrics["onnx"]["ok"]:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
