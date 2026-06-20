# archivo con todas las rutas de los archivos 

import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SOURCE_DIR = PROJECT_ROOT / "src"
ANALYZER_DIR = PROJECT_ROOT / "Analizador de video"


def _prefer_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


YOLOV2_DIR = _prefer_existing(ANALYZER_DIR / "YOLOV2", SOURCE_DIR / "YOLOV2")
YOLO_DIR = _prefer_existing(ANALYZER_DIR / "YOLO", SOURCE_DIR / "YOLO")

# El detector V2 es el modelo principal. El anterior se conserva como respaldo.
YOLOV2_WEIGHTS_PATH = YOLOV2_DIR / "best.pt"
LEGACY_YOLO_WEIGHTS_PATH = YOLO_DIR / "best.pt"
YOLO_WEIGHTS_PATH = YOLOV2_WEIGHTS_PATH

FIELD_SEGMENTATION_DIR = _prefer_existing(
    ANALYZER_DIR / "FIELD_SEGMENTATION",
    SOURCE_DIR / "FIELD_SEGMENTATION",
)
FIELD_SEGMENTATION_WEIGHTS_PATH = FIELD_SEGMENTATION_DIR / "best.pt"


def resolve_yolo_weights(
    model_version: str = "v2",
    custom_path: str | Path | None = None,
) -> Path:
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return LEGACY_YOLO_WEIGHTS_PATH if str(model_version).lower() == "legacy" else YOLOV2_WEIGHTS_PATH

LOHA_DIR = _prefer_existing(ANALYZER_DIR / "LoHa", SOURCE_DIR / "LoHa")
DORA_DIR = _prefer_existing(ANALYZER_DIR / "DoRa", SOURCE_DIR / "DoRa")
CV_MODELS_DIR = _prefer_existing(ANALYZER_DIR / "cv_models", SOURCE_DIR / "cv_models")
TRACKER_DIR = _prefer_existing(ANALYZER_DIR / "tracker", SOURCE_DIR / "tracker")


def clean_video_name (video_path: str | Path) -> str:
    '''Convierte el nombre del archivo en un nombre apto para una carpeta'''
    video_name = Path(video_path).stem
    normalized = unicodedata.normalize ("NFKD", video_name)
    ascii_name = normalized.encode ("ascii", "ignore").decode("ascii")

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", ascii_name)
    safe_name = safe_name.strip("_")

    return safe_name or "vide_sin_nombre"

def get_video_outputdirec (video_path:str | Path)-> Path:
    '''crea la carpeta y regresa output/<nombre_video>/.'''
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    video_name=clean_video_name (video_path)
    output_directory = OUTPUTS_DIR / video_name
    output_directory.mkdir(parents=True, exist_ok=True)


    return output_directory

