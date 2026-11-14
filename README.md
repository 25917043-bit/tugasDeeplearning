# The Juggling Bounces Challenge

Automatically count the number of times three basketballs bounce on the floor in a video of a
person juggling and bouncing basketballs, using deep learning.

The problem is decomposed into three sub-tasks:

1. **Detection** — locate every basketball in each frame (YOLOv8, "sports ball" class).
2. **Tracking** — keep a stable identity for each ball across frames (custom centroid tracker).
3. **Temporal event detection** — decide when a ball actually hits the floor (peak detection).

## Pipeline overview

| Stage | Method |
|---|---|
| Detection | YOLOv8s (Ultralytics), class `32` = "sports ball" on COCO |
| Tracking | Custom nearest-neighbor centroid tracker |
| Bounce counting | Local-maximum (peak) detection of the ball's bottom edge vs a floor line |

## Requirements

- Python 3.9+
- A machine with enough RAM/CPU (tested on an Apple Silicon Mac, CPU-only)

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install ultralytics opencv-python numpy
```

The first run will automatically download the YOLOv8s weights (`yolov8s.pt`).

## Usage

Run the script with an input video path:

```bash
python bounce_counter.py path/to/input_video.mp4
```

Or without arguments — it will prompt for the path:

```bash
python bounce_counter.py
```

The script writes an annotated output video (`output_demo.mp4`) with:

- Green bounding boxes around every detected ball.
- A red "Floor Threshold" line.
- A live bounce counter in the top-left corner.

## Configuration

All tunable parameters are at the top of the `process_video()` function in
[`bounce_counter.py`](bounce_counter.py):

```python
FLOOR_Y_THRESHOLD = int(height * 0.55)   # floor line position (fraction of frame height)
PEAK_WINDOW       = 3                     # window for local-maximum (peak) detection
BOUNCE_COOLDOWN   = 6                     # min frames between bounces per ball
CONF_THRESHOLD    = 0.05                  # detection confidence threshold
MATCH_RADIUS      = 200                   # max px distance to match a ball between frames
MAX_MISSED        = 30                    # frames to keep a track alive when undetected
```

- **Raise `FLOOR_Y_THRESHOLD`** (e.g. `0.70`) to require deeper contact → fewer bounces.
- **Lower `FLOOR_Y_THRESHOLD`** (e.g. `0.50`) to accept shallower contact → more bounces.
- **Lower `CONF_THRESHOLD`** to catch more marginal (blurry/fast) detections.
- **Raise `MAX_MISSED`** to bridge longer detection gaps.

## How bounce counting works

For each tracked ball, we keep a small buffer of its bottom-edge y-coordinate (`y2`). A bounce is
detected when `y2` forms a **local maximum** (the ball reaches its lowest point and reverses
direction) that crosses the floor line. A short cooldown prevents double-counting.

Peak detection is used instead of a simple "below the line" check because the ball's contact with
the floor lasts only 1–2 frames, which a median filter would smooth away.

## Ground truth

For this assignment the bounce rate was manually counted as **23 bounces per 10 seconds** over a
~20 minute video, i.e. roughly **2760 total bounces**.
