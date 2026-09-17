#!/usr/bin/env bash
# Sets up the official RbA repo + the outlier-exposure-finetuned checkpoint,
# CPU-only. Run on the Mac, not in a sandbox.
#
# The install docs want a CUDA build for MSDeformAttn, but forward() already
# falls back to a pure-PyTorch impl if the compiled op isn't there -- the only
# real blocker is the import itself raising before that fallback gets a
# chance to run. Patch below just lets that import fail quietly instead.
#
# Gets us the Swin-B checkpoint with real COCO outlier-exposure supervision
# (AP 70.8 / FPR95 6.3 on Lost & Found per the paper) instead of our current
# zero-shot Cityscapes checkpoint.
set -euo pipefail

EXT_DIR="/Volumes/BIggen/AV/external"
mkdir -p "$EXT_DIR"
cd "$EXT_DIR"

echo "== [1/6] Cloning Detectron2 and installing CPU-only =="
if [ ! -d detectron2 ]; then
  git clone https://github.com/facebookresearch/detectron2.git
fi
python3 -c "import torch" || { echo "ERROR: torch not importable -- activate the av-detector conda env first."; exit 1; }
# --no-build-isolation: detectron2's setup.py imports torch at build time,
# but pip's isolated build env doesn't have it even though our env does
pip install --no-build-isolation -e ./detectron2

echo "== [2/6] Cloning the official RbA repo (it vendors its own mask2former/ copy) =="
if [ ! -d RbA ]; then
  git clone https://github.com/NazirNayal8/RbA.git
fi
cd RbA
# requirements.txt lists "zmq" which isn't a real PyPI package (it's pyzmq),
# so the batch install fails silently -- installing the corrected list here
pip install cython scipy shapely timm h5py submitit scikit-image easydict \
  albumentations fairscale pyzmq webp ood-metrics opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git   # not on PyPI, needs git install
# mask2former's __init__ eagerly imports every backbone and mapper to
# register them, even though we only need Swin -- if another import error
# shows up, same fix: pip install whatever the traceback names

echo "== [3/6] Patching the CPU import blocker (1 line, see header comment above) =="
DEFORM_FUNC="mask2former/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py"
python3 - "$DEFORM_FUNC" <<'PYEOF'
import sys, re
path = sys.argv[1]
src = open(path).read()
needle = 'except ModuleNotFoundError as e:'
if 'MSDA = None' in src:
    print("Already patched, skipping.")
else:
    old = '''try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    info_string = (
        "\\n\\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\\n"
        "\\t`cd mask2former/modeling/pixel_decoder/ops`\\n"
        "\\t`sh make.sh`\\n"
    )
    raise ModuleNotFoundError(info_string)'''
    new = '''try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    # let this fail quietly on CPU -- forward() already falls back to the
    # pure-PyTorch path when calling into MSDA raises
    MSDA = None'''
    if old not in src:
        print("WARNING: expected text not found verbatim -- inspect this file by hand:", path)
        sys.exit(1)
    src = src.replace(old, new)
    open(path, 'w').write(src)
    print("Patched:", path)
PYEOF

echo "== [4/6] Downloading config + checkpoint (swin_b_1dl_rba_ood_coco) =="
mkdir -p ckpts/swin_b_1dl_rba_ood_coco
if [ ! -f ckpts/swin_b_1dl_rba_ood_coco/config.yaml ]; then
  curl -L -o ckpts/swin_b_1dl_rba_ood_coco/config.yaml \
    https://raw.githubusercontent.com/NazirNayal8/RbA/main/ckpts/swin_b_1dl_rba_ood_coco/config.yaml
fi
if [ ! -f ckpts/swin_b_1dl_rba_ood_coco/model_final.pth ]; then
  curl -L -o /tmp/swin_b_1dl_rba_ood_coco.zip \
    https://github.com/NazirNayal8/RbA/releases/download/model-weights/swin_b_1dl_rba_ood_coco.zip
  # the zip's own top-level folder is also named swin_b_1dl_rba_ood_coco/,
  # so unzip into ckpts/ (not ckpts/swin_b_1dl_rba_ood_coco/) or it doubles up
  unzip -o /tmp/swin_b_1dl_rba_ood_coco.zip -d ckpts
else
  echo "Checkpoint already present, skipping download."
fi

echo "== [5/6] Sanity import check (no dataset registration, just: does it import) =="
python3 -c "
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
import detectron2
print('detectron2 OK:', detectron2.__version__)
from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDA
print('MSDA is None (expected on CPU):', MSDA is None)
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
print('Mask2Former pixel decoder imports OK')
"

echo "== [6/6] Done. =="
