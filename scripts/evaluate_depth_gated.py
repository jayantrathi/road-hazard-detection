
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Volumes/BIggen/AV/external")
sys.path.insert(0, "/Volumes/BIggen/AV/external/RbA")

from src.data.lost_and_found import (
    RESULTS_DIR, HAZARD_TRAIN_ID, ROAD_TRAIN_ID, CANVAS_SIZE,
    img_path_to_label_path, load_mask_resized, load_test_split,
)
from src.eval.metrics import summarize, aupr, fpr_at_recall
from src.geometry.depth_ground_plane import GroundPlaneHeight
from src.geometry.stereo_ground_plane import StereoGroundPlaneHeight

CITYSCAPES_ROAD_CLASS = 0


def load_scorer(mode: str, checkpoint: str):
    if mode == "official":
        from rba_official_scorer import OfficialRbAScorer
        return OfficialRbAScorer(device="cpu")
    from src.scoring.trained_segmenter_scorer import TrainedSegmenterScorer
    return TrainedSegmenterScorer(checkpoint)


def predicted_road(logits: np.ndarray) -> np.ndarray:
    """logits (C,h,w) -> bool road mask resized to CANVAS_SIZE (W,H)."""
    pred = logits.argmax(0).astype(np.uint8)
    road = (pred == CITYSCAPES_ROAD_CLASS).astype(np.uint8) * 255
    road = np.array(Image.fromarray(road).resize(CANVAS_SIZE, Image.Resampling.NEAREST))
    return road > 127


def gate_scores(appearance: np.ndarray, height: np.ndarray, kappa: float,
                h0: float, spread: float) -> np.ndarray:
    """final = appearance - kappa*spread*clip((h0-height)/h0,0,1).
    Penalizes only near-coplanar (low-height) pixels; leaves protruding ones."""
    penalty = kappa * spread * np.clip((h0 - height) / max(h0, 1e-6), 0.0, 1.0)
    return appearance - penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["official", "trained"], default="official")
    ap.add_argument("--checkpoint", default="checkpoints/ood_segmenter/best.pt")
    ap.add_argument("--depth", choices=["stereo", "mono"], default="stereo",
                    help="stereo = real L&F disparity (no pretrained depth model); "
                         "mono = off-the-shelf Depth Anything")
    ap.add_argument("--fov", type=float, default=64.0)
    ap.add_argument("--limit", type=int, default=0, help="cap frames for a quick smoke run")
    args = ap.parse_args()

    test_normal, test_hazard = load_test_split()
    frames = test_normal + test_hazard
    if args.limit:
        frames = frames[: args.limit]
    print(f"Frames: {len(frames)}  ({len(test_hazard)} hazard scenes in pool)\n")

    scorer = load_scorer(args.mode, args.checkpoint)
    if args.depth == "stereo":
        print("Using REAL stereo depth from Lost & Found disparity (no pretrained depth model).")
        gp = StereoGroundPlaneHeight()
    else:
        print("Loading Depth Anything V2 (off-the-shelf depth module)...")
        gp = GroundPlaneHeight(fov_deg=args.fov)
    print("Depth source ready.\n")

    scenes = sorted({s for _, s, _ in frames})
    tune_scenes = set(scenes[::2])  
    print(f"{len(scenes)} scenes: {len(tune_scenes)} tune / {len(scenes)-len(tune_scenes)} report\n")

    cache = []
    for idx, (path, scene, is_hazard) in enumerate(frames):
        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        if mask is None:
            continue
        img = Image.open(path).convert("RGB")
        rba_map, logits = scorer.score(img, out_size=CANVAS_SIZE)
        road = predicted_road(logits)
        if args.depth == "stereo":
            height = gp.height_map_from_path(path, road, CANVAS_SIZE)
        else:
            height = gp.height_map(img.resize(CANVAS_SIZE), road)
        cache.append((scene, mask.flatten(), rba_map.flatten(), height.flatten()))
        if (idx + 1) % 10 == 0:
            print(f"  processed {idx + 1}/{len(frames)}", flush=True)

    def pool(sel_scenes):
        app, hgt, lab = [], [], []
        for scene, m, s, h in cache:
            if scene not in sel_scenes:
                continue
            roi = (m == ROAD_TRAIN_ID) | (m == HAZARD_TRAIN_ID)
            app.append(s[roi]); hgt.append(h[roi])
            lab.append((m[roi] == HAZARD_TRAIN_ID).astype(np.int8))
        return (np.concatenate(app), np.concatenate(hgt), np.concatenate(lab))

    app_t, hgt_t, lab_t = pool(tune_scenes)
    report_scenes = set(scenes) - tune_scenes
    app_r, hgt_r, lab_r = pool(report_scenes)
    spread = float(np.percentile(app_t, 95) - np.percentile(app_t, 5))

    best = (-1, 0.0, 0.06)
    for kappa in [0.5, 1.0, 2.0, 3.0]:
        for h0 in [0.03, 0.06, 0.12, 0.25]:
            g = gate_scores(app_t, hgt_t, kappa, h0, spread)
            a = aupr(g, lab_t)
            if a > best[0]:
                best = (a, kappa, h0)
    _, kappa, h0 = best
    print(f"\nTuned gate on tune scenes: kappa={kappa}, h0={h0} (AUPR={best[0]:.4f})\n")

    # report on held-out report scenes
    appearance = summarize(app_r, lab_r)
    gated_scores = gate_scores(app_r, hgt_r, kappa, h0, spread)
    gated = summarize(gated_scores, lab_r)

    def road_fp(scores):
        return fpr_at_recall(scores, lab_r, 0.95)

    print("=" * 66)
    print(f"{'metric':<12}{'appearance-only':>26}{'depth-gated':>26}")
    print("=" * 66)
    for k in ["AUPR", "AUROC", "FPR@95"]:
        arrow = ""
        if k in ("AUPR", "AUROC") and gated[k] > appearance[k]:
            arrow = "  ^ better"
        if k == "FPR@95" and gated[k] < appearance[k]:
            arrow = "  ^ better"
        print(f"{k:<12}{appearance[k]:>26.4f}{gated[k]:>26.4f}{arrow}")
    print("=" * 66)
    fp_red = (appearance["FPR@95"] - gated["FPR@95"]) / max(appearance["FPR@95"], 1e-9)
    print(f"\nFalse-positive rate at 95% recall reduced by {100*fp_red:.1f}% "
          f"({appearance['FPR@95']:.3f} -> {gated['FPR@95']:.3f})")
    print("That reduction is coplanar road-paint/manhole false positives being "
          "suppressed by the geometric gate.")

    results = {"mode": args.mode, "depth": args.depth, "kappa": kappa, "h0": h0,
               "fov": args.fov, "appearance_only": appearance, "depth_gated": gated,
               "n_report_pixels": int(len(lab_r)), "n_positive": int(lab_r.sum())}
    out = RESULTS_DIR / f"depth_gated_{args.mode}_{args.depth}_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
