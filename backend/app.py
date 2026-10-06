import io
import os
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from model import get_alzheimer_model

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Alzheimer's MRI Classifier API", version="2.0.0")

# CORS: allow_origins="*" and allow_credentials=True is an invalid combination
# that browsers reject. Since this API uses no cookies or auth headers,
# credentials are not needed — drop it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    "Cognitively Normal (CN)",
    "Mild Cognitive Impairment (MCI)",
    "Alzheimer's Disease (AD)"
]

BINARY_CLASS_NAMES = [
    "Cognitively Normal (CN)",
    "Cognitively Impaired (CDR ≥ 0.5)"
]

# ── Model Registry ────────────────────────────────────────────────────────────
# Each entry is one selectable model. The `id` is what the frontend sends
# in the `model` form field. Weight paths can be overridden with env vars.
MODEL_CONFIGS = {
    "corrected": {
        "name": "Corrected model (age-matched cohort)",
        "path": os.environ.get("CORRECTED_MODEL_PATH", "alz_resnet18_3class_v2.pt"),
        "class_names": CLASS_NAMES,
        "confidence_threshold": 0.50,
        # trained on three stacked axial slices (R, G, B = -12, 0, +12 from mid)
        "expects_stacked": True,
    },
    "binary": {
        "name": "Corrected binary model (CN vs impaired)",
        "path": os.environ.get("BINARY_MODEL_PATH", "alz_resnet18_binary_v2.pt"),
        "class_names": BINARY_CLASS_NAMES,
        # With two classes the top class is always >= 50%, so a higher bar is needed
        "confidence_threshold": 0.65,
        "expects_stacked": True,
    },
    "original": {
        "name": "Original model (age-confounded)",
        # MODEL_PATH kept for backward compatibility with the old deployment
        "path": os.environ.get("ORIGINAL_MODEL_PATH",
                               os.environ.get("MODEL_PATH", "best_alzheimer_model.pt")),
        "class_names": CLASS_NAMES,
        "confidence_threshold": 0.50,
        # trained on single grayscale slices copied into all three channels
        "expects_stacked": False,
    },
}

DEFAULT_MODEL = "corrected"

# ── Model Loading ─────────────────────────────────────────────────────────────
# Load every model whose weights are present. A missing file only disables
# that model rather than crashing the server.
loaded_models = {}
for model_id, cfg in MODEL_CONFIGS.items():
    if os.path.exists(cfg["path"]):
        try:
            # model.py handles both bare weights and v2 checkpoint dicts
            loaded_models[model_id] = get_alzheimer_model(
                model_path=cfg["path"],
                device=str(device),
                num_classes=len(cfg["class_names"]),
            )
            print(f"[ok] Loaded '{model_id}' from '{cfg['path']}'")
        except Exception as e:
            print(f"[warn] Could not load '{model_id}' from '{cfg['path']}': {e}")
    else:
        print(f"[warn] Weights for '{model_id}' not found at '{cfg['path']}' — model disabled.")

if not loaded_models:
    raise RuntimeError(
        "No model weights found. Place alz_resnet18_3class_v2.pt, "
        "alz_resnet18_binary_v2.pt and/or best_alzheimer_model.pt next to app.py, "
        "or set CORRECTED_MODEL_PATH / BINARY_MODEL_PATH / ORIGINAL_MODEL_PATH."
    )

# ── Inference Transform ───────────────────────────────────────────────────────
# Must exactly match the val_transform used during training
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Used by hosting platforms to verify the server is running."""
    return {
        "status": "ok",
        "device": str(device),
        "models_loaded": list(loaded_models.keys()),
    }


@app.get("/models")
def list_models():
    """Lists the selectable models and whether each one is available."""
    return [
        {"id": model_id, "name": cfg["name"], "available": model_id in loaded_models}
        for model_id, cfg in MODEL_CONFIGS.items()
    ]


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Choose one of: {', '.join(MODEL_CONFIGS)}"
        )
    if model not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail=f"The '{model}' model is not available on this server right now."
        )

    # Validate file type before processing
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got: {file.content_type}"
        )

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    cfg = MODEL_CONFIGS[model]
    class_names = cfg["class_names"]

    # A grayscale upload has three identical channels. The corrected models were
    # trained on three different stacked slices, so flag that mismatch.
    arr = np.asarray(img, dtype=np.int16)
    channels_identical = (np.abs(arr[..., 0] - arr[..., 1]).max() <= 2 and
                          np.abs(arr[..., 1] - arr[..., 2]).max() <= 2)
    input_mismatch = cfg["expects_stacked"] and channels_identical
    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = loaded_models[model](img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    confidence_scores = {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}
    top_class         = max(confidence_scores, key=confidence_scores.get)
    top_confidence    = float(probs.max())

    return {
        "model":                  model,
        "model_name":             cfg["name"],
        "prediction":             top_class,
        "confidence_scores":      confidence_scores,
        "low_confidence_warning": top_confidence < cfg["confidence_threshold"],
        "input_mismatch_warning": bool(input_mismatch),
        "disclaimer": (
            "Research prototype only. "
            "Not validated for clinical use."
        )
    }
