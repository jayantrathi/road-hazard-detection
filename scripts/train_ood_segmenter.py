from __future__ import annotations

import os


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.cityscapes_ood import CityscapesOOD  # noqa: E402

NUM_CLASSES = 19


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> torch.nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(
        weights=None, weights_backbone="IMAGENET1K_V2",
        num_classes=NUM_CLASSES, aux_loss=True,
    )
    return model


def outlier_loss(logits: torch.Tensor, omask: torch.Tensor) -> torch.Tensor:
    """Push sum_c tanh(logit_c) DOWN on outlier pixels -> high anomaly score.
    logits: (B,C,H,W)  omask: (B,H,W) bool. Returns 0 if no outlier pixels."""
    if omask.sum() == 0:
        return logits.sum() * 0.0 
    tanh_sum = logits.tanh().sum(dim=1)  # (B,H,W)
    return tanh_sum[omask].mean()


@torch.no_grad()
def eval_miou(model, loader, device, max_batches=40) -> float:
    """Confusion-matrix mIoU on Cityscapes val (subset, for speed)."""
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i, (x, target, _) in enumerate(loader):
        if i >= max_batches:
            break
        out = model(x.to(device))["out"]
        pred = out.argmax(1).cpu().numpy().ravel()
        gt = target.numpy().ravel()
        keep = gt != 255
        pred, gt = pred[keep], gt[keep]
        idx = gt * NUM_CLASSES + pred
        conf += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = inter / np.maximum(union, 1)
    valid = union > 0
    return float(iou[valid].mean()) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityscapes", default="data/cityscapes")
    ap.add_argument("--outliers", default="data/hazard_crops")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lambda-out", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/ood_segmenter")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))
    print(f"device={device}  epochs={args.epochs}  batch={args.batch_size}  crop={args.crop}")

    train_ds = CityscapesOOD(args.cityscapes, args.outliers, split="train",
                             crop_size=args.crop, train=True, seed=args.seed)
    val_ds = CityscapesOOD(args.cityscapes, args.outliers, split="val",
                           train=False, seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = build_model().to(device)


    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.SGD(
        [{"params": backbone_params, "lr": args.lr * 0.1},
         {"params": head_params, "lr": args.lr}],
        momentum=0.9, weight_decay=1e-4,
    )

    total_iters = args.epochs * len(train_ld)
    start_epoch = 0
    ckpt_last = out_dir / "last.pt"
    if args.resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    ce = torch.nn.CrossEntropyLoss(ignore_index=255)
    best_miou = 0.0
    it = start_epoch * len(train_ld)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run = {"ce": 0.0, "out": 0.0, "n": 0}
        for x, target, omask in train_ld:
            x, target, omask = x.to(device), target.to(device), omask.to(device)

            lr_scale = (1 - it / max(total_iters, 1)) ** 0.9
            opt.param_groups[0]["lr"] = args.lr * 0.1 * lr_scale
            opt.param_groups[1]["lr"] = args.lr * lr_scale

            out = model(x)
            logits, aux = out["out"], out["aux"]
            loss_ce = ce(logits, target) + 0.4 * ce(aux, target)
            loss_out = outlier_loss(logits, omask)
            loss = loss_ce + args.lambda_out * loss_out

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run["ce"] += loss_ce.item()
            run["out"] += loss_out.item()
            run["n"] += 1
            it += 1
            if run["n"] % 50 == 0:
                print(f"  e{epoch} it{run['n']}/{len(train_ld)} "
                      f"ce={run['ce']/run['n']:.3f} out={run['out']/run['n']:.3f} "
                      f"lr={opt.param_groups[1]['lr']:.5f}")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "args": vars(args)}, ckpt_last)

        miou = eval_miou(model, val_ld, device)
        dt = time.time() - t0
        print(f"[epoch {epoch}] ce={run['ce']/run['n']:.3f} out={run['out']/run['n']:.3f} "
              f"val_mIoU={miou:.3f}  ({dt/60:.1f} min)")
        if miou >= best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou},
                       out_dir / "best.pt")
            print(f"  -> new best mIoU {miou:.3f}, saved best.pt")

    print(f"done. best val mIoU={best_miou:.3f}. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
