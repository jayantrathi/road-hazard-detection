
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Volumes/BIggen/AV/external")
sys.path.insert(0, "/Volumes/BIggen/AV/external/RbA")

from src.eval.metrics import summarize

RA21_ROOT = Path("/Volumes/BIggen/AV/data/roadanomaly21")
RESULTS_DIR = Path("/Volumes/BIggen/AV/results")


def load_scorer(mode: str, checkpoint: str):
    if mode == "official":
        from rba_official_scorer import OfficialRbAScorer
        return OfficialRbAScorer(device="cpu")
    from src.scoring.trained_segmenter_scorer import TrainedSegmenterScorer
    return TrainedSegmenterScorer(checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["official", "trained"], default="trained")
    ap.add_argument("--checkpoint", default="checkpoints/ood_segmenter/best.pt")
    args = ap.parse_args()

    images = sorted(p for p in (RA21_ROOT / "images").glob("*.jpg")
                    if not p.name.startswith("._"))
    if not images:
        print(f"No RoadAnomaly21 images under {RA21_ROOT/'images'}")
        sys.exit(1)
    print(f"RoadAnomaly21: {len(images)} images\n")

    scorer = load_scorer(args.mode, args.checkpoint)
    print()

    all_scores, all_labels = [], []
    for idx, img_path in enumerate(images):
        lab_path = RA21_ROOT / "labels_masks" / f"{img_path.stem}_labels_semantic.png"
        if not lab_path.exists():
            continue
        label = np.array(Image.open(lab_path))
        H, W = label.shape
        img = Image.open(img_path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=(W, H))

        flat_s, flat_l = rba_map.flatten(), label.flatten()
        keep = (flat_l == 0) | (flat_l == 1)  # drop 255 ignore
        all_scores.append(flat_s[keep])
        all_labels.append(flat_l[keep].astype(np.int8))
        print(f"  scored {idx + 1}/{len(images)}", flush=True)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    res = summarize(scores, labels)

    n_pos = int(labels.sum())
    print(f"\n{len(labels):,} pixels, {n_pos:,} anomaly ({100*n_pos/len(labels):.3f}%)")
    print("\n" + "=" * 44)
    print(f"RoadAnomaly21  ({args.mode} model)")
    print("=" * 44)
    for k in ["AUPR", "AUROC", "FPR@95"]:
        print(f"  {k:<8}{res[k]:.4f}")
    print("=" * 44)
    print("Published SOTA range on RoadAnomaly21: AP ~0.5-0.9, FPR95 ~0.1-0.3")
    print("(10 public val images -> read as a generalization signal, not a leaderboard number)")

    out = RESULTS_DIR / f"roadanomaly21_{args.mode}_trained_results.json"
    out.write_text(json.dumps({"mode": args.mode, **res,
                               "n_pixels": int(len(labels)), "n_anomaly": n_pos}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
