import json
import os
from pathlib import Path
from typing import Any, Literal

import cv2
from PIL import Image

from src.shared.paths import DORA_DIR, LOHA_DIR


class SAMSegmenter:
    def __init__(
        self,
        mode: Literal["LoHa", "DoRa"] = "LoHa",
        confidence_threshold: float = 0.40,
        api_path: str | Path | None = None,
    ) -> None:
        if mode not in {"LoHa", "DoRa"}:
            raise ValueError("El modo debe ser 'LoHa' o 'DoRa'.")

        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.adapter_path = LOHA_DIR if mode == "LoHa" else DORA_DIR
        self.device = self._select_device()

        self._authenticate(api_path)
        self._load_model()

    @staticmethod
    def _select_device() -> str:
        import torch

        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @staticmethod
    def _authenticate(api_path: str | Path | None) -> None:
        from huggingface_hub import login

        token = os.getenv("HF_TOKEN")

        if not token and api_path:
            with Path(api_path).open("r", encoding="utf-8") as file:
                token = json.load(file).get("API")

        if token:
            login(token=token)

    def _load_model(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import Sam3Model, Sam3Processor

        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"No se encontro el adaptador: {self.adapter_path}"
            )

        print(f"Cargando SAM3 con adaptador {self.mode}...")
        print("La primera carga puede tardar varios minutos.")

        self.processor = Sam3Processor.from_pretrained(str(self.adapter_path))

        base_model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.float32,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path),
            is_trainable=False,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"SAM3 + {self.mode} cargado correctamente en {self.device}.")


    def detect(
        self,
        frame,
        prompt: str,
    ) -> list[dict[str, Any]]:
        import torch
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        detections = []

        for box, score in zip(results["boxes"], results["scores"]):
            detections.append({
                "source": "sam3",
                "adapter": self.mode,
                "class_name": prompt,
                "confidence": float(score.detach().cpu().item()),
                "bbox_xyxy": [
                    round(float(value), 2)
                    for value in box.detach().cpu().tolist()
                ],
            })

        return detections