import torch
import torch.nn as nn
from torchvision import models


def _extract_state_dict(checkpoint):
    """
    Accepts either a bare state_dict (old format) or a checkpoint dict
    that wraps the weights under "state_dict" (new v2 format), and
    returns plain weights. Also strips a "module." prefix left by
    DataParallel and maps a plain Linear head ("fc.weight") onto the
    Dropout + Linear head ("fc.1.weight") used here.
    """
    meta = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        checkpoint = checkpoint["state_dict"]

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key in ("fc.weight", "fc.bias"):
            key = key.replace("fc.", "fc.1.")
        state_dict[key] = value
    return state_dict, meta


def get_alzheimer_model(model_path=None, device='cpu', num_classes=3):
    """
    Initializes the ResNet-18 architecture with the custom Dropout +
    Linear classifier head used during training.

    Args:
        model_path  : path to a saved .pt file (bare weights or a
                      checkpoint dict), or None for an untrained model.
        device      : 'cpu' or 'cuda'
        num_classes : number of output classes (3 for CN / MCI / AD,
                      2 for the binary model)

    Returns:
        model in eval mode, moved to `device`
    """
    model = models.resnet18()
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict, meta = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        print(f"Weights loaded from: {model_path}")
        if meta.get("classes") is not None:
            # Printed so the label order can be checked against app.py
            print(f"  checkpoint class order: {meta['classes']}")

    return model.to(device).eval()
