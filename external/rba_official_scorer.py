
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict

RBA_REPO = Path("/Volumes/BIggen/AV/external/RbA")
if str(RBA_REPO) not in sys.path:
    sys.path.insert(0, str(RBA_REPO))

CONFIG_PATH = RBA_REPO / "ckpts" / "swin_b_1dl_rba_ood_coco" / "config.yaml"
MODEL_PATH = RBA_REPO / "ckpts" / "swin_b_1dl_rba_ood_coco" / "model_final.pth"


class OfficialRbAScorer:
    def __init__(self, device: str = "cpu"):
        # local imports -- must happen after RBA_REPO is on sys.path, and
        # after cwd/env is set up by the caller (see module docstring)
        from train_net import Trainer, setup
        from detectron2.checkpoint import DetectionCheckpointer

        self.device = torch.device(device)
        args = edict({
            "config_file": str(CONFIG_PATH),
            "eval-only": True,
            # MODEL.DEVICE must be overridden here, not just in our own
            # .to(self.device) call below -- detectron2's build_model()
            # reads cfg.MODEL.DEVICE and calls model.to(torch.device(...))
            # internally, BEFORE we ever get the model back. The checkpoint's
            # saved config.yaml defaults this to "cuda" (it was trained on a
            # GPU cluster), which crashes immediately on a CPU-only torch
            # build with "Torch not compiled with CUDA enabled" -- has to be
            # forced to cpu at the config level, not patched after the fact.
            "opts": ["OUTPUT_DIR", "output/", "MODEL.DEVICE", "cpu"],
        })
        self.cfg = setup(args)
        self.model = Trainer.build_model(self.cfg)
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            str(MODEL_PATH), resume=False
        )
        self.model.to(self.device)
        self.model.eval()

        # read straight from the checkpoint's own config instead of assuming
        self.input_format = self.cfg.INPUT.FORMAT  # "RGB" or "BGR"
        print(f"Loaded official RbA (Swin-B, COCO outlier supervision). "
              f"INPUT.FORMAT={self.input_format}, device={self.device}")

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB")).astype(np.float32)  # H,W,3 RGB, 0-255
        if self.input_format == "BGR":
            arr = arr[:, :, ::-1].copy()
        # Detectron2 models normalize internally (pixel_mean/std baked into
        # the model's own preprocessing) -- we hand over raw 0-255 values,
        # channel-first, NOT manually normalized here.
        tensor = torch.as_tensor(arr.transpose(2, 0, 1))  # 3,H,W
        return tensor

    def score(self, img: Image.Image, out_size: tuple[int, int] | None = None):
        """Returns (rba_map: np.ndarray[H,W], raw_logits or None) -- same
        shape as src/scoring/mask2former_rba.py's RbAScorer.score() so this
        drops into the existing eval harness."""
        orig_w, orig_h = img.size
        tensor = self._to_tensor(img)

        with torch.no_grad():
            out = self.model([{"image": tensor.to(self.device)}])
            logits = out[0]["sem_seg"]  # (19, H, W) -- native model resolution
            rba = -logits.tanh().sum(dim=0)  # (H, W), higher = more anomalous

        rba_np = rba.cpu().numpy()

        if out_size is not None and (rba_np.shape[1], rba_np.shape[0]) != tuple(out_size):
            rba_img = Image.fromarray(rba_np.astype(np.float32), mode="F")
            rba_img = rba_img.resize(out_size, Image.Resampling.BILINEAR)
            rba_np = np.array(rba_img)

        return rba_np, logits.cpu().numpy()


if __name__ == "__main__":
    # Minimal sanity check: load the model, score one real Lost & Found
    # frame, print basic stats (min/max/mean, any NaNs) -- run this BEFORE
    # wiring into the full eval harness, so a broken load fails fast and
    # cheap instead of silently producing garbage across 30+ frames.
    import glob
    scorer = OfficialRbAScorer(device="cpu")

    laf_root = Path("/Volumes/BIggen/AV/data/lost_and_found")
    candidates = list(laf_root.rglob("*_leftImg8bit.png"))
    if not candidates:
        print(f"No test image found under {laf_root} -- pass a path manually.")
        sys.exit(1)
    test_img_path = candidates[0]
    print(f"Scoring test image: {test_img_path}")

    img = Image.open(test_img_path).convert("RGB")
    rba_map, _ = scorer.score(img, out_size=(1200, 675))

    print(f"rba_map shape: {rba_map.shape}")
    print(f"min={rba_map.min():.4f} max={rba_map.max():.4f} mean={rba_map.mean():.4f} "
          f"std={rba_map.std():.4f} has_nan={np.isnan(rba_map).any()}")
    print("\nIf this printed sane numbers (no NaN, non-degenerate spread), "
          "the load worked -- next step is wiring this into evaluate_rba_lost_and_found.py.")
