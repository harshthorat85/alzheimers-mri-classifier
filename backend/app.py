import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from model import get_alzheimer_model

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Alzheimer's MRI Classifier API", version="1.0.0")

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

# ── Model Loading ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = os.environ.get("MODEL_PATH", "best_alzheimer_model.pt")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model weights not found at '{MODEL_PATH}'. "
        "Download best_alzheimer_model.pt from Google Drive and place it "
        "in the same directory as app.py, or set the MODEL_PATH env variable."
    )

model = get_alzheimer_model(model_path=MODEL_PATH, device=str(device))

# ── Inference Transform ───────────────────────────────────────────────────────
# Must exactly match the val_transform used during training
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    "Cognitively Normal (CN)",
    "Mild Cognitive Impairment (MCI)",
    "Alzheimer's Disease (AD)"
]

CONFIDENCE_THRESHOLD = 0.50

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Used by hosting platforms to verify the server is running.
    Also confirms the model loaded successfully.
    """
    return {
        "status": "ok",
        "device": str(device),
        "model": "ResNet-18 fine-tuned on OASIS-1"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type before processing
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got: {file.content_type}"
        )

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    confidence_scores = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(3)}
    top_class         = max(confidence_scores, key=confidence_scores.get)
    top_confidence    = float(probs.max())

    return {
        "prediction":            top_class,
        "confidence_scores":     confidence_scores,
        "low_confidence_warning": top_confidence < CONFIDENCE_THRESHOLD,
        "disclaimer": (
            "Research prototype only. "
            "Not validated for clinical use."
        )
    }
