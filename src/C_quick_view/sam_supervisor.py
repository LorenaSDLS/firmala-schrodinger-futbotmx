from pathlib import Path
from typing import Any


class SAMSegmenter:
    def __init__(
        self,
        model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.is_loaded = False

    def load(self) -> None:
        """
        Aqui vamos a cargar SAM/SAM3 cuando confirmemos:
        1. nombre exacto del modelo base
        2. carpeta exacta del adaptador LoHa o DoRa
        3. si se carga con transformers, segment-anything o una clase custom
        """
        print("SAM todavia esta en modo placeholder.")
        self.is_loaded = False

    def segment_frame(self, frame) -> list[dict[str, Any]]:
        """
        Regresara mascaras o cajas refinadas.
        Por ahora no hace nada para que YOLO funcione primero.
        """
        return []