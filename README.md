<div align="center">

# Road-Hazard Corner-Case Detection

Flagging unknown obstacles on the road — the debris, lost cargo, and animals a self-driving car's perception was never trained to recognize.

![hero](docs/images/hero_tricycle.jpg)

*Input with the true hazard in green · raw anomaly heat on the road · after the depth gate, with the alert box.*

</div>

## About

Most self-driving perception models are trained on a fixed set of classes — car, person, road, sign. Anything outside that set gets confidently mislabeled instead of flagged, which is a problem when the thing outside the set is a tire in your lane or a pothole left open.

This is an anomaly detector for that gap. A segmentation network is trained to become *unconfident* on unfamiliar objects, so wherever it's unsure on the road, that's a likely hazard. A depth-based filter then removes false alarms from flat road markings and manhole covers. The whole model was trained from scratch, no cloud, no pretrained anomaly checkpoint.

**Built with:** PyTorch (MPS) · DeepLabV3-ResNet50 · Cityscapes · Lost & Found · RoadAnomaly21

## Results

Lost & Found, road-region protocol, scene-level held-out split:

| Method | AUPR | AUROC | FPR@95 |
|---|---|---|---|
| Trained model | 0.69 | 0.96 | 0.22 |
| **+ depth gate** | **0.89** | **0.98** | **0.11** |
| Swin-B checkpoint *(reference)* | 0.78 | — | 0.29 |
| RbA paper *(reference)* | 0.70 | — | 0.06 |

The trained model gets pretty much the same average precision as the published paper, 0.69 vs. 0.70. Adding the depth gate also pushes it past the pretrained Swin-B checkpoint on both AUPR and false positive rate.

On the second benchmark, RoadAnomaly21, the AUPR drops to 0.37. That makes sense since the model is built more specifically for hazards on the road, rather than detecting every kind of unusual object or scene.

## How it works

1 Train a DeepLabV3-ResNet50 on Cityscapes, while adding random unknown objects into the scenes so the model learns to be less confident when it sees something unfamiliar.
2 Give each pixel an anomaly score based on how poorly it matches any of the known classes.
3 Only look for anomalies inside the predicted drivable road area, while ignoring the vehicle and its parts (hood on most dashcams).
4 Depth gate — estimate the 3D plane of the road and filter out things that are basically flat with the road, like paint or manhole covers. Objects that actually stick up from the road are kept since they are more likely to be real hazards.
5 Draw boxes around the remaining detections so the final hazards are easy to see.

![depth gate](docs/images/result_markings.jpg)

*Road markings and a manhole firing (middle) get knocked down by the depth gate (right), while the real hazard is still showcased.*

## Limitations

* It was trained mostly on paved Cityscapes roads, so it tends to flag gravel and other unfamiliar surfaces too often.
* The false positive rate is a bit higher than what the paper reports.
* Right now, depth comes from a single camera. Lost & Found also provides real stereo disparity, so that could be an easy upgrade later.
* The test set is only 30 frames, so small differences probably aren’t that meaningful.

## Getting started

```bash
pip install -r requirements.txt
# Cityscapes goes in data/cityscapes, Lost & Found in data/lost_and_found
```

## Usage

```bash
# train the segmenter 
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_ood_segmenter.py --epochs 25

# evaluate on Lost & Found, and the depth-gated version
python scripts/evaluate_trained_ood.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/evaluate_depth_gated.py --mode trained --depth mono

# generate the demo strips 
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/demo_trained_pipeline.py --n 30
```

## Contact

Jayant Rathi — jayant12rathi@gmail.com

