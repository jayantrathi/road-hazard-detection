
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Volumes/BIggen/AV/external")
sys.path.insert(0, "/Volumes/BIggen/AV/external/RbA")

from src.data.lost_and_found import (
    RESULTS_DIR, HAZARD_TRAIN_ID, CANVAS_SIZE,
    img_path_to_label_path, load_mask_resized, load_test_split,
)
from evaluate_depth_gated import predicted_road, gate_scores
from src.scoring.trained_segmenter_scorer import TrainedSegmenterScorer
from src.geometry.depth_ground_plane import GroundPlaneHeight

OUT_DIR = RESULTS_DIR / "trained_demo"
KAPPA, H0 = 1.0, 0.03           # gate params tuned in evaluate_depth_gated.py
ALERT_PCT = 99.0                # alert threshold: top-1% of gated score on road
MIN_ALERT_AREA = 30
PANEL_W = 460                   # per-panel display width

# --- drivable-road corridor cleaning (ported from predict_hazards_demo.py) ---
# The method operates on the drivable road, not the whole frame -- so known
# objects (cars, buildings) outside the road are not searched, and the ego
# vehicle's own hood/ornament is excluded (it otherwise fires every frame).
CITYSCAPES_ROAD_CLASS = 0
HOOD_BAND_FRAC = 0.87
EGO_ORNAMENT_ROW = (0.74, 1.00)
EGO_ORNAMENT_COL = (0.36, 0.64)
EGO_DILATE_ITERS = 18
ROAD_ERODE_ITERS = 10


def clean_road_corridor(logits: np.ndarray) -> np.ndarray:
    """Predicted road, cleaned into the region the system actually searches:
    ego vehicle excluded, holes filled, boundary eroded to kill the seam."""
    pred = logits.argmax(axis=0).astype(np.uint8)
    road = (pred == CITYSCAPES_ROAD_CLASS).astype(np.uint8) * 255
    road = np.array(Image.fromarray(road).resize(CANVAS_SIZE, Image.Resampling.NEAREST)) > 127
    road = ndimage.binary_fill_holes(road)
    h, w = road.shape
    ego = np.zeros_like(road)
    ego[int(h * HOOD_BAND_FRAC):, :] = True
    ego[int(h * EGO_ORNAMENT_ROW[0]):int(h * EGO_ORNAMENT_ROW[1]),
        int(w * EGO_ORNAMENT_COL[0]):int(w * EGO_ORNAMENT_COL[1])] = True
    ego = ndimage.binary_dilation(ego, iterations=EGO_DILATE_ITERS)
    road &= ~ego
    if ROAD_ERODE_ITERS:
        road = ndimage.binary_erosion(road, iterations=ROAD_ERODE_ITERS)
    return road


def heat_overlay(base_rgb: np.ndarray, score: np.ndarray, region: np.ndarray) -> Image.Image:
    """Red-hot overlay of `score`, tinted ONLY inside `region` (the drivable
    corridor). Everything outside the corridor is dimmed, so the viewer sees
    that the system searches the road, not the whole frame."""
    try:  # matplotlib >= 3.9
        from matplotlib import colormaps
        cmap = colormaps["inferno"]
    except Exception:  # older matplotlib
        from matplotlib import cm
        cmap = cm.get_cmap("inferno")
    roi = region if region.any() else np.ones_like(region, bool)
    # Floor at the 85th percentile of corridor scores: the bulk of the road
    # (below typical) stays dark, only genuinely elevated anomaly lights up, so
    # a discrete hazard pops instead of the whole surface glowing.
    lo, hi = np.percentile(score[roi], 85), np.percentile(score[roi], 99.5)
    norm = np.clip((score - lo) / (hi - lo + 1e-9), 0, 1)
    heat = (cmap(norm) * 255).astype(np.uint8)[:, :, :3]
    a = (norm[:, :, None] * 0.85)
    tinted = (a * heat + (1 - a) * base_rgb).astype(np.uint8)
    dimmed = (base_rgb.astype(np.float32) * 0.5).astype(np.uint8)  # outside corridor
    out = np.where(region[:, :, None], tinted, dimmed)
    return Image.fromarray(out.astype(np.uint8))


def alert_boxes(gated: np.ndarray, road: np.ndarray):
    roi_vals = gated[road] if road.any() else gated.ravel()
    thr = np.percentile(roi_vals, ALERT_PCT)
    mask = (gated >= thr) & road
    lbl, n = ndimage.label(mask)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(ys) < MIN_ALERT_AREA:
            continue
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max(), len(ys)))
    boxes.sort(key=lambda b: -b[4])
    return boxes[:3]


def caption(img: Image.Image, text: str) -> Image.Image:
    bar_h = 26
    canvas = Image.new("RGB", (img.width, img.height + bar_h), (18, 18, 18))
    canvas.paste(img, (0, bar_h))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    d.text((8, 5), text, fill=(235, 235, 235), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ood_segmenter/best.pt")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, test_hazard = load_test_split()
    frames = test_hazard[: args.n]
    print(f"Rendering {len(frames)} demo frames\n")

    scorer = TrainedSegmenterScorer(args.checkpoint)
    gp = GroundPlaneHeight()
    print()

    strips = []
    W, H = CANVAS_SIZE
    for idx, (path, scene, is_hazard) in enumerate(frames):
        img = Image.open(path).convert("RGB")
        base = np.array(img.resize(CANVAS_SIZE))
        rba, logits = scorer.score(img, out_size=CANVAS_SIZE)
        road = clean_road_corridor(logits)   # ego-excluded, eroded drivable corridor
        height = gp.height_map(img.resize(CANVAS_SIZE), road)
        spread = float(np.percentile(rba[road], 95) - np.percentile(rba[road], 5)) if road.any() else 1.0
        gated = gate_scores(rba, height, KAPPA, H0, spread)

        # panel 1: original (+ GT hazard outline in green)
        p1 = Image.fromarray(base.copy())
        mask = load_mask_resized(img_path_to_label_path(path))
        d1 = ImageDraw.Draw(p1)
        if mask is not None:
            gt = (mask == HAZARD_TRAIN_ID)
            edge = gt ^ ndimage.binary_erosion(gt, iterations=2)
            ov = np.array(p1); ov[edge] = [0, 255, 0]; p1 = Image.fromarray(ov)

        # panel 2: appearance heatmap
        p2 = heat_overlay(base, rba, road)
        # panel 3: depth-gated heatmap + alert boxes + GT outline
        p3 = heat_overlay(base, gated, road)
        d3 = ImageDraw.Draw(p3)
        if mask is not None:
            ov = np.array(p3); ov[edge] = [0, 255, 0]; p3 = Image.fromarray(ov); d3 = ImageDraw.Draw(p3)
        for (x0, y0, x1, y1, _) in alert_boxes(gated, road):
            d3.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3)

        # compose strip
        def rs(im):
            return im.resize((PANEL_W, int(PANEL_W * H / W)))
        p1c = caption(rs(p1), "input  (green = true hazard)")
        p2c = caption(rs(p2), "appearance anomaly (raw model)")
        p3c = caption(rs(p3), "depth-gated + alert boxes")
        strip = Image.new("RGB", (p1c.width * 3 + 16, p1c.height), (18, 18, 18))
        strip.paste(p1c, (0, 0)); strip.paste(p2c, (p1c.width + 8, 0)); strip.paste(p3c, (2 * p1c.width + 16, 0))
        out = OUT_DIR / f"demo_{idx:02d}_{scene}.jpg"
        strip.save(out, quality=90)
        strips.append(strip)
        print(f"  saved {out.name}")

    # contact sheet (stack strips)
    if strips:
        cw = strips[0].width
        ch = sum(s.height for s in strips) + 8 * (len(strips) - 1)
        sheet = Image.new("RGB", (cw, ch), (18, 18, 18))
        y = 0
        for s in strips:
            sheet.paste(s, (0, y)); y += s.height + 8
        sheet.save(OUT_DIR / "contact_sheet.jpg", quality=88)
        print(f"\nsaved contact_sheet.jpg")
        # animated GIF of just the depth-gated alert panels
        gif_frames = [s.crop((2 * (PANEL_W) + 16, 0, s.width, s.height)) for s in strips]
        gif_frames[0].save(OUT_DIR / "demo.gif", save_all=True, append_images=gif_frames[1:],
                           duration=900, loop=0)
        print(f"saved demo.gif")
    print(f"\nAll demo visuals in {OUT_DIR}")
    print("LOOK at demo_00..09: the middle panel should show the raw model firing on")
    print("both the hazard and some road paint; the right panel should show the paint")
    print("suppressed and red boxes on the real hazard. That's the before/after story.")


if __name__ == "__main__":
    main()
