# archivo con todas las rutas de los archivos 

import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ANALYZER_DIR = PROJECT_ROOT / "Analizador de video"

YOLO_DIR = ANALYZER_DIR / "YOLO"
YOLO_WEIGHTS_PATH = YOLO_DIR / "best.pt"

LOHA_DIR = ANALYZER_DIR / "LoHa"
DORA_DIR = ANALYZER_DIR / "DoRa"
CV_MODELS_DIR = ANALYZER_DIR / "cv_models"
TRACKER_DIR = ANALYZER_DIR / "tracker"


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

