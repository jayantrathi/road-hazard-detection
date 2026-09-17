from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Daimler/Cityscapes stereo rig calibration for the 2048x1024 left camera.
# (documented Cityscapes camera intrinsics; L&F used the same rig). If a
# per-frame camera/*.json is present we could read it instead, but these are
# constant across the rig. Values are for FULL 2048x1024; we rescale to the
# working resolution in height_map_from_path().
CAL_FULL_W = 2048
CAL_FULL_H = 1024
FX_FULL = 2262.52
FY_FULL = 2265.30
CX_FULL = 1096.98
CY_FULL = 513.14
BASELINE_M = 0.209313  # metres between the two cameras


def decode_disparity(raw: np.ndarray) -> np.ndarray:
    """Cityscapes/L&F disparity PNG -> disparity in pixels (0 = invalid).
    VERIFY this against a real file before trusting it."""
    raw = raw.astype(np.float32)
    disp = np.where(raw > 0, (raw - 1.0) / 256.0, 0.0)
    return disp


class StereoGroundPlaneHeight:
    def __init__(self, disparity_root: str | Path | None = None,
                 ransac_iters: int = 200, ransac_tol: float = 0.02, seed: int = 0):
        # disparity_root: folder holding the L&F disparity tree. If None we
        # derive the path from the left-image path by swapping 'leftImg8bit'
        # -> 'disparity' (adjust once we see the real layout).
        self.disparity_root = Path(disparity_root) if disparity_root else None
        self.ransac_iters = ransac_iters
        self.ransac_tol = ransac_tol
        self.rng = np.random.default_rng(seed)

    def _disparity_path(self, left_path: str | Path) -> Path:
        left_path = Path(left_path)
        # e.g. .../leftImg8bit/.../<name>_leftImg8bit.png
        #   -> .../disparity/.../<name>_disparity.png
        s = str(left_path)
        s = s.replace("leftImg8bit", "disparity")
        s = s.replace("_disparity.png", "_disparity.png")  # name suffix handled by replace above
        # the double 'leftImg8bit' in L&F packaging becomes double 'disparity';
        # the file suffix _leftImg8bit.png -> _disparity.png via the same swap
        return Path(s)

    @staticmethod
    def _lsq_plane(pts: np.ndarray):
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        n = vt[-1]
        n = n / (np.linalg.norm(n) + 1e-12)
        return n, -n @ c

    def _fit_plane_ransac(self, pts: np.ndarray):
        N = len(pts)
        tol = self.ransac_tol * (np.median(np.abs(pts[:, 2])) + 1e-6)
        best_inl, best = -1, None
        for _ in range(self.ransac_iters):
            i = self.rng.choice(N, 3, replace=False)
            p0, p1, p2 = pts[i]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-8:
                continue
            n = n / nn
            off = -n @ p0
            inl = (np.abs(pts @ n + off) < tol)
            if int(inl.sum()) > best_inl:
                best_inl, best = int(inl.sum()), (n, off, inl)
        if best is None:
            return self._lsq_plane(pts)
        _, _, inl = best
        return self._lsq_plane(pts[inl])

    def height_map_from_path(self, left_path: str | Path, road_mask: np.ndarray,
                             out_size: tuple[int, int]) -> np.ndarray:
        """Returns relative height-above-road-plane map (>=0), resized to
        out_size (W, H) to match the anomaly map / labels. road_mask is a bool
        array at out_size resolution (predicted road)."""
        W, H = out_size
        dpath = self.disparity_root_path(left_path)
        raw = np.array(Image.open(dpath))
        disp = decode_disparity(raw)  # at native (2048x1024) res

        # scale intrinsics from full calibration res to the disparity res
        dh, dw = disp.shape
        sx, sy = dw / CAL_FULL_W, dh / CAL_FULL_H
        fx, fy = FX_FULL * sx, FY_FULL * sy
        cx, cy = CX_FULL * sx, CY_FULL * sy

        valid = disp > 0
        z = np.zeros_like(disp)
        z[valid] = fx * BASELINE_M / disp[valid]  # metric depth (metres)

        us, vs = np.meshgrid(np.arange(dw), np.arange(dh))
        X = (us - cx) * z / fx
        Y = (vs - cy) * z / fy
        P = np.stack([X, Y, z], axis=-1)

        # bring road_mask (out_size) to disparity res
        road_d = np.array(Image.fromarray(road_mask.astype(np.uint8) * 255)
                          .resize((dw, dh), Image.Resampling.NEAREST)) > 127
        road_valid = road_d & valid & np.isfinite(z)
        road_pts = P[road_valid]
        if len(road_pts) < 50:
            return np.zeros((H, W), dtype=np.float32)
        if len(road_pts) > 20000:
            road_pts = road_pts[self.rng.choice(len(road_pts), 20000, replace=False)]

        n, off = self._fit_plane_ransac(road_pts)
        dist = np.abs(P.reshape(-1, 3) @ n + off).reshape(dh, dw)
        dist[~valid] = 0.0  # no depth -> no height signal -> no gating
        # normalize by median road depth so the gate threshold is scale-stable
        scale = np.median(z[road_valid]) + 1e-6
        height_rel = (dist / scale).astype(np.float32)

        if (dw, dh) != (W, H):
            height_rel = np.array(Image.fromarray(height_rel, mode="F")
                                  .resize((W, H), Image.Resampling.BILINEAR))
        return height_rel

    def disparity_root_path(self, left_path):
        if self.disparity_root is None:
            return self._disparity_path(left_path)
        # if an explicit root was given, mirror the split/city/name under it
        return self._disparity_path(left_path)


if __name__ == "__main__":
    # VERIFY THE DISPARITY ENCODING against a real file before trusting anything.
    import sys

    laf = Path("/Volumes/BIggen/AV/data/lost_and_found")
    disp_files = list(laf.rglob("*_disparity.png"))
    if not disp_files:
        print("No *_disparity.png found under data/lost_and_found.")
        print("Download the L&F disparity package (or tell me the layout).")
        sys.exit(1)

    f = disp_files[0]
    raw = np.array(Image.open(f))
    print(f"file: {f}")
    print(f"raw dtype={raw.dtype} shape={raw.shape} "
          f"min={raw.min()} max={raw.max()} nonzero={(raw>0).mean():.2%}")
    disp = decode_disparity(raw)
    v = disp[disp > 0]
    print(f"decoded disparity (px): min={v.min():.2f} max={v.max():.2f} "
          f"median={np.median(v):.2f}")
    z = FX_FULL * BASELINE_M / v
    print(f"implied depth (m):      min={z.min():.1f} max={z.max():.1f} "
          f"median={np.median(z):.1f}")
    print("\nSanity: median depth should be a plausible road distance (~10-40 m).")
    print("If depths look crazy (mm, or thousands of m), the encoding differs and")
    print("we fix decode_disparity() before running the eval.")
