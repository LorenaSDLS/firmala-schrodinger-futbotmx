from cv_models.SAM3 import SAM3
from random import choice, seed
from pathlib import Path

def load_all_images(folder_path: str):
    video_folder_path = Path(folder_path)
    videos_path_list = video_folder_path.rglob("*.jpg")
    return list(videos_path_list)

gerente = SAM3("api_sam3.json", 
               ruta_loha="LoHa", 
               ruta_dora="DoRA",
               mode="LoHa")

gerente._load_model(mode="DoRa", conf_threshold=0.40)

image_list = load_all_images("Val_dataset/val_images/")
seed(42)
single_image_path = choice(image_list)

gerente.load_image(single_image_path, show_image=True)

print("Haciendo Boxes para el campo")
gerente.make_image_boxes(prompt="playing field")

print("Visualizando las cajas para campo")
gerente.visualize_image_boxes()

print("Haciendo Boxes para pelota")
gerente.make_image_boxes(prompt="orange ball")

print("Visualizando las cajas para pelota")
gerente.visualize_image_boxes()

print("Haciendo Boxes para robot")
gerente.make_image_boxes(prompt="robots")

print("Visualizando las cajas para robot")
gerente.visualize_image_boxes()