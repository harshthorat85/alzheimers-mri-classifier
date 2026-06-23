# Alzheimer's MRI Classifier

ResNet-18 fine-tuned on OASIS-1 to classify 2D axial MRI slices into:
- **CN** — Cognitively Normal
- **MCI** — Mild Cognitive Impairment  
- **AD** — Alzheimer's Disease

## Repo Structure

```
alzheimers-mri-classifier/
├── model.py                  # Architecture definition
├── app.py                    # FastAPI inference server
├── requirements.txt          # Python dependencies
└── best_alzheimer_model.pt   # Trained weights (add manually — not in git)
```

> **Note:** `best_alzheimer_model.pt` is not committed to this repo (it's ~45MB).
> Download it from Google Drive and place it here before running.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

API will be live at `http://localhost:8000`.  
Swagger docs at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server + model status check |
| POST | `/predict` | Upload an MRI slice, get diagnosis probabilities |

## Disclaimer

Research prototype trained on ~300 OASIS-1 samples.  
Not validated for clinical use.