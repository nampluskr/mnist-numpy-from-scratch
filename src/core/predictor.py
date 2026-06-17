# predictor.py: Task-aware prediction post-processing for raw model logits.

import numpy as np
from src.nn.activations import sigmoid


class Predictor:
    def __init__(self, model, task_spec):
        self.model = model
        self.mode = task_spec["prediction_mode"]

    def predict(self, images):
        """Returns {"logits", "predictions"} for input images.

        predictions:
            argmax     → class index (N,) int32
            threshold  → 0/1 label  (N,) int32
            round_clip → digit 0~9  (N,) int32
        """
        logits = self.model.forward(images)
        if self.mode == "argmax":
            predictions = logits.argmax(axis=1).astype(np.int32)
        elif self.mode == "threshold":
            predictions = (sigmoid(logits) >= 0.5).astype(np.int32).ravel()
        else:  # round_clip
            predictions = np.round(np.clip(logits * 9.0, 0, 9)).astype(np.int32).ravel()
        return {"logits": logits, "predictions": predictions}
