from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "ml"
FRONTEND_DIR = ROOT_DIR / "frontend"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config_utils import default_config_path, load_config, resolve_ml_path  # noqa: E402
from predict import load_model_and_classes, transform  # noqa: E402


app = FastAPI(title="Greek Letters ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_MODEL_CACHE: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "model": None,
    "classes": None,
}


def _load_app_config() -> tuple[dict[str, Any], str]:
    config, config_path = load_config(default_config_path())
    return config, config_path


def _model_path_from_config(config: dict[str, Any]) -> str:
    paths_cfg = config.get("paths", {})
    model_path = resolve_ml_path(paths_cfg.get("model_path", "models/best_model.pth"))
    return os.path.abspath(model_path)


def _plot_path_from_config(config: dict[str, Any]) -> str:
    paths_cfg = config.get("paths", {})
    plot_path = resolve_ml_path(paths_cfg.get("plot_path", "plots/architecture_comparison.png"))
    return os.path.abspath(plot_path)


def _get_model_and_classes(model_path: str):
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    mtime = os.path.getmtime(model_path)
    cache_hit = (
        _MODEL_CACHE["path"] == model_path
        and _MODEL_CACHE["mtime"] == mtime
        and _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["classes"] is not None
    )
    if cache_hit:
        return _MODEL_CACHE["model"], _MODEL_CACHE["classes"]

    model, classes = load_model_and_classes(model_path)
    _MODEL_CACHE["path"] = model_path
    _MODEL_CACHE["mtime"] = mtime
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["classes"] = classes
    return model, classes


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    config, config_path = _load_app_config()
    model_path = _model_path_from_config(config)

    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model_cfg = checkpoint.get("config", {})

    return {
        "config_path": config_path,
        "model_path": model_path,
        "model_last_updated": os.path.getmtime(model_path),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_score": checkpoint.get("selection_score"),
        "selected_seed": checkpoint.get("selected_seed"),
        "seeds": checkpoint.get("seeds", []),
        "num_classes": len(checkpoint.get("classes", [])),
        "architecture": {
            "channels": model_cfg.get("channels"),
            "hidden_size": model_cfg.get("hidden_size"),
            "dropout": model_cfg.get("dropout"),
        },
    }


@app.get("/api/architecture-comparison")
def architecture_comparison_image() -> FileResponse:
    config, _ = _load_app_config()
    plot_path = _plot_path_from_config(config)

    if not os.path.isfile(plot_path):
        raise HTTPException(status_code=404, detail=f"Architecture comparison plot not found: {plot_path}")

    return FileResponse(plot_path, media_type="image/png")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    config, _ = _load_app_config()
    model_path = _model_path_from_config(config)

    model, classes = _get_model_and_classes(model_path)

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    model_device = next(model.parameters()).device
    image_tensor = transform(image).unsqueeze(0).to(model_device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)

    confidence, predicted_idx = torch.max(probs, 0)
    topk = min(3, len(classes))
    top_probs, top_indices = torch.topk(probs, k=topk)

    top_predictions = [
        {
            "label": classes[idx.item()],
            "confidence": float(prob.item() * 100),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "prediction": classes[predicted_idx.item()],
        "confidence": float(confidence.item() * 100),
        "top_predictions": top_predictions,
    }
