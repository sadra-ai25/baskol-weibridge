# Weighbridge Monitoring System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey) ![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-red) ![Docker](https://img.shields.io/badge/Docker-Compose-blue)

AI-powered weighbridge monitoring system that validates vehicle presence on a scale using computer vision. Sends a base64-encoded camera snapshot to the API and receives a validation result confirming whether a vehicle is correctly positioned on the weighbridge.

## Features

- **Automatic vehicle detection** — YOLOv11 detects vehicles and validates their position on the scale platform
- **Multi-weighbridge support** — configurable per-weighbridge ROI polygons (wb1, wb3)
- **Base64 image API** — accepts images as base64 strings or file uploads (up to 32 MB)
- **ROI polygon validation** — checks whether the detected vehicle overlaps with the weighbridge platform region
- **Batch processing** — batch endpoint for processing multiple images at once
- **Annotation output** — returns annotated images showing detections and ROI overlays
- **CLI tools** — helper scripts to define and update ROI regions from sample images

## Tech Stack

| Component | Technology |
|---|---|
| AI Model | YOLOv11 (Ultralytics) |
| API Server | Flask |
| Image Processing | OpenCV, NumPy |
| Containerization | Docker Compose |

## Architecture

```
Weighbridge Scale Camera
         │
         │  (snapshot)
         ▼
  Weighbridge Software  ──POST /validate──▶  Flask API
                                                  │
                                            ObjectDetector (YOLOv11)
                                                  │  detect vehicles
                                            SimpleValidator
                                                  │  check ROI polygon
                                                  ▼
                                           JSON Response
                                      { valid: true/false,
                                        vehicle_detected: true,
                                        annotated_image: "base64..." }
```

## Prerequisites

- Docker & Docker Compose
- YOLOv11 model weights at `weights/best.pt`
- Weighbridge ROI config at `configs/weighbridges.json`

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/sadra-ai25/baskol-weibridge.git
cd baskol-weibridge

# 2. Place model weights
mkdir -p weights
cp /path/to/best.pt weights/

# 3. Configure weighbridge ROIs
mkdir -p configs
# Use the CLI tool to define ROI from a sample image:
python tools/extract_polygon_points.py --image sample/wb1_sample.jpg

# 4. Start the service
docker compose up -d --build
```

## ROI Configuration

Define the weighbridge platform as a polygon in `configs/weighbridges.json`:

```json
{
  "wb1": {
    "roi_polygon": [[100, 200], [800, 200], [800, 600], [100, 600]],
    "camera_id": "cam_wb1"
  },
  "wb3": {
    "roi_polygon": [[150, 180], [820, 180], [820, 580], [150, 580]],
    "camera_id": "cam_wb3"
  }
}
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/validate` | Validate vehicle presence from image |
| `POST` | `/batch` | Batch validate multiple images |
| `GET` | `/health` | Service health check |

### Example: Validate via base64

```bash
curl -X POST http://localhost:4001/validate \
  -H "Content-Type: application/json" \
  -d '{
    "weighbridge_id": 1,
    "image": "<base64-encoded-image>"
  }'
```

### Response

```json
{
  "valid": true,
  "vehicle_detected": true,
  "vehicle_on_scale": true,
  "confidence": 0.91,
  "annotated_image": "<base64-encoded-annotated-image>"
}
```

## CLI Tools

```bash
# Define ROI polygon interactively from image
python tools/extract_polygon_points.py --image sample/wb1.jpg

# Update ROI config from a new reference image
python tools/update_roi_from_image.py --wb wb1 --image sample/new_wb1.jpg

# Crop and inspect ROI region
python tools/crop_roi_cli.py --wb wb1 --image sample/wb1.jpg
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
