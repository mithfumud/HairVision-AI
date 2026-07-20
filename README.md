# HairVision-AI

HairVision-AI looks at photos of someone's scalp and estimates how much hair loss is showing — and roughly where they sit on the Norwood scale.

You can upload a **front** photo, a **crown** (top-of-head) photo, or both. The system finds the areas where hair *should* be, compares that to where hair actually is, and turns the gap into simple clinical-style metrics.

This repo includes a **very basic Streamlit UI** so people can try the pipeline without wiring up an API. It's meant as a demo / MVP showcase, not a polished consumer product.

Under the hood it uses **SegFormer face-parsing** ([`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing)) for hair/skin/head masks and **MediaPipe Face Landmarker** for front-face geometry.

---

## What it does

1. **Checks the photo**  
   Is it usable? Too small? Blurry? For front images, is there a face?  
   Soft issues (like lower-than-recommended resolution) become warnings and analysis still runs. Tiny/broken images are blocked.

2. **Segments hair and head**  
   Separates hair from scalp/head so later steps aren't guessing from raw pixels alone.

3. **Builds a "normative" region**  
   - Front: expected hair-bearing zone from face landmarks / hairline geometry  
   - Crown: expected crown envelope around the vertex / whorl area  

4. **Finds the deficit**  
   Where the normative region says hair should be, but coverage is missing or thinning is visible.

5. **Reports numbers + Norwood stage**  
   Loss percentages by zone (front, temples, crown), overall coverage, and a Norwood classification with a short explanation.

---

## Models used

HairVision-AI doesn’t train a new model from scratch. It combines two off-the-shelf vision models with classical computer-vision logic on top:

### 1. Face / hair segmentation — SegFormer face parsing
- **Model:** [`jonathandinu/face-parsing`](https://huggingface.co/jonathandinu/face-parsing)
- **Architecture:** SegFormer (via Hugging Face `transformers`)
- **What it’s for:** Semantic segmentation of the photo into classes like hair, skin, ears, etc. (CelebAMask-HQ-style labels)
- **How we use it:** Build hair, skin, and head masks that drive deficit analysis and metrics

### 2. Face landmarks — MediaPipe Face Landmarker
- **Model:** MediaPipe **Face Landmarker** (`face_landmarker.task`)
- **What it’s for:** 3D face landmarks on front-view photos
- **How we use it:** Confirm a face is present, validate front vs crown views, and place the frontal hairline / normative scalp region

### Everything else
Norwood staging, crown vertex finding, coverage refinement, and deficit math are **rule-based / classical CV** on top of those masks — not a separate deep-learning classifier.

---

## Demo UI (Streamlit)

`app.py` is a simple upload → analyze → results screen:

- Upload front and/or crown images (JPG/PNG)
- Hit **Analyze Hair**
- See annotated overlays, a metrics table, and a short clinical interpretation

That’s intentionally basic — just enough to showcase the backend.

### Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL (usually `http://localhost:8501`).

**Tip:** Use Python **3.11** if you can. Some vision deps are happier there than on newer versions.

---

## Project layout

| Path | What it is |
|------|------------|
| `app.py` | Basic Streamlit demo UI |
| `analysis/` | Quality checks, normative regions, deficit, metrics, Norwood, pipeline |
| `models/` | Segmentation / model wrappers |
| `test_images/` | Sample front & crown photos |
| `tests/` | Unit / QA scripts |
| `packages.txt` | System libs for Streamlit Cloud (OpenGL / GLib, etc.) |

---

## Notes

- Upload **at least one** image (front, crown, or both).
- Recommended size is **512×512 or larger**. Smaller usable images get upscaled automatically with a warning.
- Face Landmarker weights download into `assets/models/` on first run if missing.
- This is a research / MVP prototype — not a medical diagnosis tool.

---

## Disclaimer

For educational and demo purposes. Hair-loss staging from photos has limits (lighting, hair length, camera angle). Always treat outputs as approximate.
