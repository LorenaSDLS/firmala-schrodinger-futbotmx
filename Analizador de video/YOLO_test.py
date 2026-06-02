from cv_models.YOLO import YouOnlyLiveOnce
from pathlib import Path
from random import choice, seed

def load_all_images(folder_path: str):
    video_folder_path = Path(folder_path)
    videos_path_list = video_folder_path.rglob("*.jpg")
    return list(videos_path_list)

image_list = load_all_images("Val_dataset/val_images/")
seed()
single_image_path = choice(image_list)

yolo = YouOnlyLiveOnce(yolo_pt_path="/mnt/d/Documentos/FutBot/firmala-schrodinger-futbotmx/runs/detect/train/weights/best.pt")

print("Cargando la imagen a YOLO")
yolo.load_image(single_image_path, show_image=True)
yolo.analyze_frame()

print("Visualizando las cajitas con YOLO")
yolo.visualize_image_boxes()