from ultralytics import YOLO as UltralyticsYOLO 
import torch
from typing import Literal, Optional
import numpy as np
from PIL.Image import Resampling
from pathlib import Path
from PIL import Image

class YouOnlyLiveOnce(): 
    def __init__(self, yolo_pt_path: str):
        self.yolo_path = yolo_pt_path
        self.model = None # Es buena práctica declararlo vacío al inicio
        self._load_model()

    def _load_model(self, device: Literal["cuda", "cpu"] = None):
        # Arreglada la indentación de este bloque
        if not hasattr(self, 'model') or self.model is None:
            if not device:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device not in ["cuda", "cpu"]:
                raise ValueError(f"Error fatal: El dispositivo '{device}' no es válido. Usa 'cpu' o 'cuda'.")
            else:
                self.device = device
                
            print(f"Cargando YOLO a {self.device}.... ")
            # Usamos el nombre seguro que importamos arriba
            self.model = UltralyticsYOLO(self.yolo_path) 
            self.model.to(self.device)
            print(f"Se cargó YOLO exitosamente a {self.device}")
        else:
            print(f"Modelo ya cargado a {self.device}")

    # Es mejor usar np.ndarray en el typing
    def analyze_frame(self, frame: Optional[np.ndarray] = None):
        # Usamos .track() explícitamente y apagamos los prints molestos de la consola con verbose=False

        if frame is None:
            # Verificamos por seguridad que sí haya usado load_image() antes
            if hasattr(self, 'imagen_pil') and self.imagen_pil is not None:
                frame = np.asarray(self.imagen_pil)
            else:
                raise ValueError("Error: No se pasó ningún frame y no hay ninguna imagen cargada en memoria.")

        resultados_brutos = self.model.track(source=frame, persist=True, verbose=False)
        
        # ¡Clave! Extraemos el primer resultado de la lista
        resultado_actual = resultados_brutos[0] 
        
        self.mis_resultados_json = []

        # Iteramos caja por caja dentro de las detecciones de ESE resultado
        # Es importante revisar que sí haya detectado algo
        if resultado_actual.boxes is not None:
            for box in resultado_actual.boxes:
                # 1. Extraemos los datos del tensor
                confianza = box.conf[0].item()             
                clase_id = int(box.cls[0].item())          
                coordenadas = box.xyxy[0].tolist()         
                
                # 2. Obtenemos el nombre de la clase
                nombre_clase = resultado_actual.names[clase_id] 
                
                # 3. Sacamos el ID único del objeto (si no hay, le ponemos -1)
                tracking_id = int(box.id[0].item()) if box.id is not None else -1

                # 4. Armamos el diccionario
                mi_diccionario = {
                    "confidence": confianza,
                    "class_name": nombre_clase,
                    "class_id": clase_id,
                    "tracking_id": tracking_id, 
                    "box": [int(c) for c in coordenadas] # [x_min, y_min, x_max, y_max]
                }
                
                self.mis_resultados_json.append(mi_diccionario)

        # En producción usamos return en lugar de print
        return self.mis_resultados_json

    def load_image(self, frame_or_path, show_image: bool = False, target_size: tuple = (512, 512),
                   resampling: Resampling = Resampling.BILINEAR):
        
        # 1. Si es un string (texto) O un objeto Path (de pathlib)
        if isinstance(frame_or_path, (str, Path)):
            # Lo convertimos a string forzosamente para que PIL lo entienda
            self.imagen_pil = Image.open(str(frame_or_path)).convert("RGB")
            
        # 2. Si es un frame de video de OpenCV
        elif isinstance(frame_or_path, np.ndarray):
            import cv2
            frame_rgb = cv2.cvtColor(frame_or_path, cv2.COLOR_BGR2RGB)
            self.imagen_pil = Image.fromarray(frame_rgb)
            
        # 3. Si ya es una imagen PIL directamente
        else:
            self.imagen_pil = frame_or_path.convert("RGB")

        self.ancho, self.alto = self.imagen_pil.size
        self.imagen = self.imagen_pil.resize(target_size, resampling)
        
        if show_image:
            temp_path = "debug_imagen.jpg"
            self.imagen.save(temp_path)
            
            # Magia: Le pedimos al Windows real que la abra
            import os
            os.system(f"explorer.exe {temp_path}")

    def visualize_image_boxes(self):
        # 1. Imports hasta arriba para evitar errores de UnboundLocalError
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import time

        img_dibujada = self.imagen_pil.copy() 
        ancho_real, alto_real = img_dibujada.size
        
        draw = ImageDraw.Draw(img_dibujada) 

        colores = {
            "ball" : "#B37D2B",
            "field": "#FFFFFF",
            "robot": "#1443DE"
        }
        
        for obj in self.mis_resultados_json:
            clase = obj["class_name"]
            conf = obj["confidence"]
            caja = obj["box"]

            # 🚨 ARREGLO 1: Desempaquetamos las coordenadas
            x1, y1, x2, y2 = caja 

            # .get() ayuda por si acaso YOLO detecta algo que no esté en el diccionario de colores
            color = colores.get(clase, "#00FF00") 
            
            draw.rectangle(caja, outline=color, width=4)

            # 🚨 ARREGLO 2: Usamos la variable correcta "clase"
            texto = f"{clase}: {conf:.2f}"
            y_text = max(0, y1 - 20)
            
            # Dibujamos el recuadro del texto (quitamos el 'alpha' porque PIL no lo soporta directamente en rectángulos)
            draw.rectangle([x1, y_text, x1 + 100, y_text + 20], fill=color)
            
            # Texto negro
            draw.text((x1 + 2, y_text + 2), texto, fill="black")
            
        # ==========================================
        # 4. GUARDADO DE LA IMAGEN FINAL
        # ==========================================
        # Generamos un nombre único usando los segundos del reloj para que VS Code no nos engañe
        nombre_unico = f"resultado_{int(time.time())}.jpg"
        
        img_dibujada.save(nombre_unico)
        
        print(f"📸 ¡Éxito! Imagen real guardada como '{nombre_unico}'.")
        print(f"👆 Busca EXACTAMENTE ese archivo en el explorador de VS Code y ábrelo.")