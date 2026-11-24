import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).resolve().parent / "model"
CONFIG_PATH = MODEL_DIR / "anaemia_model_config.json"
SCALER_PATH = MODEL_DIR / "anaemia_scaler.pkl"
CHECKPOINT_PATH = MODEL_DIR / "anaemia_pytorch_model.pth"
EXPECTED_FEATURES = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]


class PyTorchInferenceError(RuntimeError):
    """Raised when the configured PyTorch model cannot produce a prediction."""


class AnaemiaNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], output_size),
        )

    def forward(self, inputs):
        return self.network(inputs)


@lru_cache(maxsize=1)
def _load_model_bundle():
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        features = config.get("features")
        hidden_layers = config.get("hidden_layers")
        threshold = config.get("threshold")
        if features != EXPECTED_FEATURES or config.get("input_size") != len(EXPECTED_FEATURES):
            raise ValueError("PyTorch model configuration has an unsupported feature schema")
        if not isinstance(hidden_layers, list) or len(hidden_layers) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in hidden_layers
        ):
            raise ValueError("PyTorch model configuration has invalid hidden layers")
        if config.get("output_size") != 1:
            raise ValueError("PyTorch checkpoint requires a single-logit output")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("PyTorch model configuration has an invalid threshold")

        scaler = joblib.load(SCALER_PATH)
        scaler_features = getattr(scaler, "feature_names_in_", None)
        if scaler_features is None or scaler_features.tolist() != features:
            raise ValueError("PyTorch scaler feature names do not match model configuration")
        if getattr(scaler, "n_features_in_", None) != len(features):
            raise ValueError("PyTorch scaler feature count does not match model configuration")

        model = AnaemiaNN(config["input_size"], hidden_layers, config["output_size"])
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.to("cpu")
        model.eval()
        return config, scaler, model
    except Exception as exc:
        raise PyTorchInferenceError("The PyTorch prediction model is unavailable.") from exc


def predict_pytorch(feature_values):
    """Scale named clinical inputs and return a binary prediction payload."""
    try:
        config, scaler, model = _load_model_bundle()
        feature_frame = pd.DataFrame(
            [[feature_values[name] for name in config["features"]]],
            columns=config["features"],
        )
        scaled_features = scaler.transform(feature_frame)
        input_tensor = torch.as_tensor(scaled_features, dtype=torch.float32, device="cpu")

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits)

        if tuple(probability.shape) != (1, 1):
            raise ValueError("PyTorch model returned an invalid output shape")
        positive_probability = float(probability.item())
        if not np.isfinite(positive_probability) or not 0 <= positive_probability <= 1:
            raise ValueError("PyTorch model returned an invalid probability")

        prediction = int(positive_probability >= config["threshold"])
        probabilities = np.array([1.0 - positive_probability, positive_probability])
        return {
            "prediction_index": prediction,
            "prediction_proba": probabilities,
            "features": feature_frame.to_numpy(),
            "features_scaled": np.asarray(scaled_features),
        }
    except PyTorchInferenceError:
        raise
    except Exception as exc:
        raise PyTorchInferenceError("The PyTorch model could not process this prediction.") from exc