# HairVision-AI

AI-powered hair loss analysis from front and/or crown scalp photos. Estimates thinning regions and Norwood stage via a Streamlit MVP.

## Features

- Front and crown image upload
- Soft quality warnings with automatic upscaling for sub-512 images
- Clinical deficit overlays, loss metrics, and Norwood classification
- Streamlit UI for demos

## Requirements

- Python 3.10+
- See `requirements.txt` (PyTorch, MediaPipe, Streamlit, OpenCV, etc.)

## Setup

```bash
cd HairVision-AI
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Face Landmarker weights download automatically into `assets/models/` on first run if missing.

## Run (Streamlit)

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit MVP UI |
| `analysis/` | Quality, normative regions, deficit, metrics, Norwood, pipeline |
| `models/` | Segmentation wrappers |
| `test_images/` | Sample front/crown photos |
| `tests/` | Unit and QA scripts |

## Notes

- Upload at least one image (front, crown, or both).
- Recommended resolution is 512×512+; smaller usable images are upscaled with a warning.
- Do not commit secrets or large model binaries (see `.gitignore`).
