import json
from pathlib import Path

class Model:
    def __init__(self, path: str, version: str):
        self.version = version
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        
        if data.get("version") != version:
            raise ValueError(f"Model version mismatch: expected {version}, got {data.get('version')}")
        
        self.w1 = float(data["w1"])
        self.w2 = float(data["w2"])
        self.b = float(data["b"])

    def predict(self, x1: float, x2: float) -> tuple[int, float]:
        # Простая линейная модель: score = w1*x1 + w2*x2 + b
        score = self.w1 * x1 + self.w2 * x2 + self.b
        prob = 1.0 / (1.0 + 2.718281828459045 ** (-score))  # Sigmoid
        label = 1 if prob >= 0.5 else 0
        return label, round(prob, 4)
