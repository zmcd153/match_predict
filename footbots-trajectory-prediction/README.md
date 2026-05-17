# FootBots Trajectory Prediction

A soccer video pipeline for player/ball detection, tracking, field-coordinate conversion, and short-horizon trajectory prediction.

## Features

- Extract frames for annotation
- Train a custom YOLO player/ball detector
- Convert video detections into field-coordinate tracks
- Clean/interpolate tracks
- Predict future trajectories with a Transformer-style model
- Render debug videos, field-board videos, and live rolling prediction overlays

## Project Structure

```text
.
├─ main.py                         # end-to-end video/tracks prediction entry point
├─ train_model.py                  # trajectory model training utilities
├─ requirements.txt
├─ homography.example.json         # example center-circle calibration
├─ models/                         # optional local model weights
├─ examples/                       # small sample files
└─ src/
   ├─ build_yolo_dataset.py
   ├─ train_yolo.py
   ├─ check_yolo.py
   ├─ render_detection_debug.py
   ├─ render_prediction_video.py
   ├─ render_live_prediction_video.py
   ├─ render_field_video.py
   └─ ...
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare Homography

Copy the example and edit the image points for your video:

```powershell
Copy-Item homography.example.json homography.json
```

For accurate overlays, calibrate with real pitch line points. The included example is only an approximate center-circle calibration.

## Train YOLO

After preparing a YOLO dataset:

```powershell
python -m src.train_yolo `
  --dataset-dir dataset_cvat `
  --model yolov8s.pt `
  --imgsz 1280 `
  --epochs 100 `
  --batch 8 `
  --project runs\detect `
  --name soccer_topview_cvat
```

## Run Prediction on Video

```powershell
python main.py `
  --video path\to\match.mp4 `
  --weights models\yolo_soccer_topview_cvat_best.pt `
  --detector-mode custom `
  --checkpoint models\metrica_game1.pt `
  --homography homography.json `
  --out runs\video_predictions.csv `
  --conf 0.12 `
  --min-length 10
```

## Debug Detection Points

```powershell
python -m src.render_detection_debug `
  --video path\to\match.mp4 `
  --raw-tracks runs\video_raw_tracks.csv `
  --out runs\video_detection_debug.mp4
```

## Render Live Rolling Predictions

```powershell
python -m src.render_live_prediction_video `
  --video path\to\match.mp4 `
  --tracks runs\video_clean_tracks.csv `
  --checkpoint models\metrica_game1.pt `
  --out runs\video_predictions_live_arrow.mp4 `
  --predict-every 5 `
  --min-visible 8 `
  --max-agents 10 `
  --pred-steps 20
```

Use `--prediction-scale 5` if the future displacement is too small to see clearly.

## Notes

- Do not commit large raw videos, datasets, or generated run outputs.
- The bundled model weights are optional and can be replaced with your own trained files.
- Better YOLO labels, especially for the ball and false-positive grass regions, usually improve results more than longer training.
