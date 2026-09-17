
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lost_and_found import (
    RESULTS_DIR, HAZARD_TRAIN_ID, ROAD_TRAIN_ID, CANVAS_SIZE,
    img_path_to_label_path, load_mask_resized, load_test_split,
)
from src.eval.metrics import summarize
from src.scoring.trained_segmenter_scorer import TrainedSegmenterScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ood_segmenter/best.pt")
    args = ap.parse_args()

    test_normal, test_hazard = load_test_split()
    frames = test_normal + test_hazard
    print(f"Held-out test frames: {len(test_normal)} normal + {len(test_hazard)} hazard "
          f"= {len(frames)}\n")

    scorer = TrainedSegmenterScorer(args.checkpoint)
    print()

    full_scores, full_labels = [], []
    roi_scores, roi_labels = [], []
    label_hist = {}

    for idx, (path, scene, is_hazard) in enumerate(frames):
        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        if mask is None:
            continue
        img = Image.open(path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)

        fs, fm = rba_map.flatten(), mask.flatten()
        for v in np.unique(fm):
            label_hist[int(v)] = label_hist.get(int(v), 0) + int((fm == v).sum())

        full_scores.append(fs)
        full_labels.append((fm == HAZARD_TRAIN_ID).astype(np.int8))
        roi = (fm == ROAD_TRAIN_ID) | (fm == HAZARD_TRAIN_ID)
        roi_scores.append(fs[roi])
        roi_labels.append((fm[roi] == HAZARD_TRAIN_ID).astype(np.int8))

        if (idx + 1) % 10 == 0:
            print(f"  scored {idx + 1}/{len(frames)}", flush=True)

    full_scores = np.concatenate(full_scores)
    full_labels = np.concatenate(full_labels)
    roi_scores = np.concatenate(roi_scores)
    roi_labels = np.concatenate(roi_labels)

    full = summarize(full_scores, full_labels)
    roi = summarize(roi_scores, roi_labels)

    print("\n" + "=" * 62)
    print(f"{'metric':<10}{'FULL FRAME':>22}{'ROI ONLY (standard)':>22}")
    print("=" * 62)
    for k in ["AUPR", "AUROC", "FPR@95"]:
        print(f"{k:<10}{full[k]:>22.4f}{roi[k]:>22.4f}")
    print("=" * 62)
    print("\nReference points:")
    print("  Downloaded Swin-B checkpoint: ROI AUPR ~0.78, FPR95 ~0.29")
    print("  RbA paper on Fishyscapes L&F: AP ~0.70, FPR95 ~0.06")
    print("  ^ The trained model above is directly comparable to these.")

    results = {"checkpoint": args.checkpoint, "full_frame": full, "roi_only": roi,
               "label_histogram": label_hist,
               "n_positive": int(roi_labels.sum()), "n_roi_pixels": len(roi_labels)}
    out = RESULTS_DIR / "trained_ood_roi_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
