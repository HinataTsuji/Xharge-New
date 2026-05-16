# ☀️ Solar PV Layout Optimizer

**Malaysian Rooftop PV Design Tool** — A Streamlit web app for optimising solar panel placement on rooftops, built with real Malaysian irradiance data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

- 📸 **Image Upload** — Upload rooftop satellite/drone images
- 🔄 **Orientation Adjustment** — Rotate, flip, and align images
- 📏 **Scale Setting** — Two-point calibration or manual pixels-per-meter
- 🔷 **Roof Boundary Drawing** — Click-to-draw polygon vertices
- 🚫 **Obstacle Marking** — Mark AC units, skylights, vents as exclusion zones
- ⚡ **Automated Optimization** — Grid-based panel placement with offset trials
- 📊 **Energy Yield Estimation** — 17 Malaysian locations with real PSH data
- 💰 **Financial Analysis** — RM savings at RM 0.571/kWh tariff
- 🌿 **CO₂ Savings** — Based on Malaysia grid emission factor (0.585 tCO₂/MWh)
- 📥 **Export Results** — Download optimization results as JSON

## 🧠 New: AI Solar Planning Backend (FastAPI)

This repository now includes a production-style backend service in `solar_backend/` for AI-assisted roof analysis and panel estimation.

### Backend Endpoints

- `GET /health` — Service health
- `GET /model-info` — Model capabilities + fine-tuning and dataset recommendations
- `POST /analyze-roof` — Upload rooftop image, detect roof/obstacles, estimate usable area, output overlay PNG (`backend`: `auto|classical|qwen_vl|sam2|sam2_qwen`)
- `POST /estimate-panels` — Compute panel layout and energy estimates from roof geometry (`placement_backend`: `raster|opencv`)

### Backend Architecture

```
solar_backend/
├── main.py
├── api/
│   ├── router.py
│   └── routes/
│       ├── health.py
│       └── analysis.py
├── core/
│   ├── config.py
│   ├── exceptions.py
│   └── logging.py
├── schemas/
│   └── api.py
├── services/
│   ├── model_registry.py
│   ├── roof_analysis_service.py
│   └── panel_estimation_service.py
├── training/
│   └── lora_finetune.py
├── pipelines/
│   ├── preprocess.py
│   ├── segmentation.py
│   ├── segmentation_qwen.py
│   ├── segmentation_sam2.py
│   ├── segmentation_fusion.py
│   ├── polygon_extraction.py
│   ├── postprocess.py
│   ├── placement.py
│   ├── placement_opencv.py
│   └── visualization.py
└── utils/
    ├── image.py
    └── geometry.py
```

### Run Backend Locally

```bash
pip install -r requirements.txt
uvicorn solar_backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t ai-solar-backend .
docker run --rm -p 8000:8000 ai-solar-backend
```

### Docker Compose (GPU)

```bash
docker compose up --build
```

Optional environment variables:
- `MODEL_DEVICE` (default `cuda`)
- `QWEN_VL_MODEL` (primary model override, tried first)
- `QWEN_VL_MODEL_CANDIDATES` (comma-separated fallback model IDs/paths)
- `QWEN_VL_REVISION` (optional revision/commit pin for reproducible model loading)
- `QWEN_VL_LOCAL_FILES_ONLY` (`true|false`, load only local cache/snapshots)
- `QWEN_VL_MODEL_LOAD_TIMEOUT_SECONDS` (HF Hub download timeout, default `30`)
- `QWEN_VL_MODEL_LOAD_ETAG_TIMEOUT_SECONDS` (HF Hub metadata timeout, default `10`)
- `QWEN_LORA_ADAPTER` (path to mounted LoRA adapter directory)
- `QWEN_LORA_MERGE` (`true|false`)

Model loading strategy:
- The backend always tries `QWEN_VL_MODEL` first.
- It then tries each entry from `QWEN_VL_MODEL_CANDIDATES` in order until one fully loads.
- If all candidates fail, Qwen backend is marked unavailable (`qwen_ready=false`) and classical CV fallback remains active.
- When safetensors artifacts are unavailable for a candidate, loading automatically retries with non-safetensors weights.

Recommended reliability workflow when `.safetensors` is missing or network is unstable:
1. Download and cache at least one known-good model snapshot in advance.
2. Set `QWEN_VL_LOCAL_FILES_ONLY=true` in production/offline environments.
3. Set `QWEN_VL_MODEL` to your preferred local path/ID, and `QWEN_VL_MODEL_CANDIDATES` to additional local snapshots.
4. Pin `QWEN_VL_REVISION` where possible for reproducible startup.
5. Tune `QWEN_VL_MODEL_LOAD_TIMEOUT_SECONDS` and `QWEN_VL_MODEL_LOAD_ETAG_TIMEOUT_SECONDS` for your network conditions.

### Example Inference Script

```bash
python scripts/example_inference.py --image /path/to/roof.png --base-url http://127.0.0.1:8000
```

### LoRA Fine-Tuning Scaffold (Qwen2.5-VL)

```bash
pip install peft accelerate
python -m solar_backend.training.lora_finetune --dataset-path /path/to/train.jsonl --output-dir artifacts/qwen-lora
```

Fine-tuning supports the same fallback and reliability controls:
- `--model-name` (primary)
- `--model-candidates` (fallback list)
- `--model-revision`
- `--model-local-files-only`
- `--model-load-timeout-seconds`
- `--model-load-etag-timeout-seconds`

Then start backend with adapter:

```bash
QWEN_LORA_ADAPTER=artifacts/qwen-lora uvicorn solar_backend.main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 Panel Specification

| Parameter | Value |
|-----------|-------|
| Model | Generic 620Wp |
| Dimensions | 2278 × 1134 mm |
| Efficiency | 21.3% |
| Temp Coefficient | -0.35 %/°C |

## 🏙️ Supported Locations (17 cities)

Kuala Lumpur, Petaling Jaya, Shah Alam, George Town, Johor Bahru, Ipoh, Melaka, Kuantan, Kota Bharu, Kuala Terengganu, Alor Setar, Seremban, Kota Kinabalu, Kuching, Putrajaya, Miri, Sandakan

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/solar-pv-optimizer.git
cd solar-pv-optimizer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
solar-pv-optimizer/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
├── rooftop_3d_midas.py       # Bonus: MiDaS depth estimation script
└── utils/
    ├── __init__.py
    ├── models.py             # Data classes (Point, Panel, Result, etc.)
    ├── geometry.py           # Polygon math, bounding box, inset
    ├── irradiance.py         # Malaysian PSH data, yield calculation
    └── optimization.py       # Panel placement optimization algorithm
```

## 🔧 How It Works

### Optimization Algorithm
1. **Edge Setback** — Polygon is inset by the configured setback distance
2. **Grid Placement** — Panels are placed in a regular grid within the inset boundary
3. **Offset Trials** — 6×6 = 36 grid offset positions are tested to find maximum packing
4. **Collision Detection** — Each panel is checked against boundary (point-in-polygon) and obstacles (AABB)
5. **Best Result** — The offset yielding the most panels is selected

### Performance Ratio Model
```
PR = (1 - soiling) × (1 - wiring) × inverter_eff × (1 - degradation) × (1 - temp_loss) × tilt_factor
```

| Component | Value |
|-----------|-------|
| Soiling Loss | 2% |
| Wiring Loss | 2% |
| Inverter Efficiency | 96% |
| Degradation | 0.5% |
| Temperature Derating | Based on NOCT model |
| Tilt Factor | Quadratic penalty from optimal (latitude) |

### Annual Yield
```
Yield (kWh) = Capacity (kWp) × PSH × 365 × PR
```

## 🐍 Bonus: 3D Roof Mesh Generator

The `rooftop_3d_midas.py` script uses **MiDaS depth estimation** + **Open3D** to generate a 3D mesh from rooftop photos.

```bash
pip install torch torchvision opencv-python open3d
python rooftop_3d_midas.py --image path/to/roof.jpg --model DPT_Large
```

## 📄 License

MIT License — free for personal and commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
