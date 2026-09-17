from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES = 19


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TrainedSegmenterScorer:
    def __init__(self, checkpoint: str | Path, device: str | None = None,
                 infer_long_side: int = 1024):
        from torchvision.models.segmentation import deeplabv3_resnet50

        self.device = torch.device(device) if device else _pick_device()
        self.infer_long_side = infer_long_side  # downscale for MPS memory/speed

        self.model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES, aux_loss=True)
        state = torch.load(str(checkpoint), map_location="cpu")
        sd = state["model"] if "model" in state else state
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()
        miou = state.get("miou")
        print(f"Loaded trained DeepLabV3 from {checkpoint} "
              f"(epoch={state.get('epoch','?')}, val_mIoU={miou if miou else '?'}), device={self.device}")

    def _preprocess(self, img: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        w, h = img.size
        long = max(w, h)
        if long > self.infer_long_side:
            s = self.infer_long_side / long
            w2, h2 = int(round(w * s)), int(round(h * s))
            img = img.resize((w2, h2), Image.Resampling.BILINEAR)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
        return t, img.size  # (w,h) of the (possibly resized) input

    @torch.no_grad()
    def score(self, img: Image.Image, out_size: tuple[int, int] | None = None):
        """Returns (rba_map: np.ndarray[H,W], logits: np.ndarray[C,H,W]).
        out_size is (W, H) to match the eval harness's CANVAS_SIZE."""
        t, _ = self._preprocess(img)
        logits = self.model(t.to(self.device))["out"][0]  # (C,h,w)
        rba = -logits.tanh().sum(dim=0)  # (h,w), higher = more anomalous
        rba_np = rba.float().cpu().numpy()
        logits_np = logits.float().cpu().numpy()

        if out_size is not None and (rba_np.shape[1], rba_np.shape[0]) != tuple(out_size):
            rba_img = Image.fromarray(rba_np, mode="F").resize(out_size, Image.Resampling.BILINEAR)
            rba_np = np.array(rba_img)
        return rba_np, logits_np


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ood_segmenter/best.pt"
    scorer = TrainedSegmenterScorer(ckpt)
    laf = Path("/Volumes/BIggen/AV/data/lost_and_found")
    cands = list(laf.rglob("*_leftImg8bit.png"))
    if not cands:
        print("No Lost & Found image found; pass one manually.")
        sys.exit(1)
    img = Image.open(cands[0]).convert("RGB")
    rba, _ = scorer.score(img, out_size=(1200, 675))
    print(f"rba shape {rba.shape} min={rba.min():.3f} max={rba.max():.3f} "
          f"mean={rba.mean():.3f} nan={np.isnan(rba).any()}")
    print("Sane spread + no NaN => model loads and scores. Next: run the ROI eval.")
