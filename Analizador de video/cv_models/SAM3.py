import torch
import gc
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
import matplotlib.pyplot as plt
from huggingface_hub import login
from contextlib import nullcontext
import json
import numpy as np
from peft import PeftModel
from pathlib import Path
from PIL.Image import Resampling
from typing import Literal
import time

class SAM3():
    def __init__(self, api_path: str, ruta_dora: str = None, ruta_loha: str = None, mode: Literal["LoHa", "DoRa"] = "DoRa"):
        """
        Inicializa la clase, limpia la memoria, se loguea en Hugging Face y carga el modelo.
        """
        torch.cuda.empty_cache()
        gc.collect()

        with open(api_path, "r") as f:
            key = json.load(f)["API"]
        login(token=key)
        
        if mode not in ["LoHa", "DoRa"]:
            raise ValueError(f"Error fatal: El modo '{mode}' no es válido. Usa 'LoHa' o 'DoRa'.")
        
        self.mode = mode
        self.ruta_loha = ruta_loha
        self.ruta_dora = ruta_dora

        self._load_model(self.mode)

    def change_mode(self, mode: Literal["LoHa", "DoRa"] = "DoRa"):
        """Permite cambiar entre los entrenamientos de LoHa y DoRa en caliente."""
        if mode not in ["LoHa", "DoRa"]:
            raise ValueError(f"Error fatal: El modo '{mode}' no es válido. Usa 'LoHa' o 'DoRa'.")

        if self.mode == mode:
            print(f"SAM3 ya en el modo de entrenamiento {mode}")
        else:
            self._load_model(mode)
        
    def _load_model(self, mode: Literal["LoHa", "DoRa"] = "DoRa", conf_threshold: float = 0.25):
        """
        Carga el modelo en la memoria (CPU o GPU). Solo se ejecuta una vez al inicio.
        """
        torch.cuda.empty_cache()
        gc.collect()
        
        if mode not in ["LoHa", "DoRa"]:
            raise ValueError(f"Error fatal: El modo '{mode}' no es válido. Usa 'LoHa' o 'DoRa'.")

        if self.mode == "LoHa":
            self.ruta = self.ruta_loha
        elif self.mode == "DoRa":
            self.ruta = self.ruta_dora

        if 'sam' not in globals():
            print("Cargando el modelo por primera vez...")
            self.device = "cpu" 
            
            self.conf_threshold = conf_threshold
            self.processor = Sam3Processor.from_pretrained(self.ruta)
            self.base_model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float32)
            
            self.model = PeftModel.from_pretrained(self.base_model, self.ruta)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Modelo cargado exitosamente en {self.device}")
        else:
            self.conf_threshold = conf_threshold
            print(f"Se cambió el confidence threshold a {conf_threshold}")

    def load_image(self, frame_or_path, show_image: bool = False, target_size: tuple = (512, 512),
                   resampling: Resampling = Resampling.BILINEAR):
        """Toma una ruta de imagen o un frame de video y lo prepara para la IA."""
        if isinstance(frame_or_path, (str, Path)):
            self.imagen_pil = Image.open(str(frame_or_path)).convert("RGB")
        elif isinstance(frame_or_path, np.ndarray):
            import cv2
            frame_rgb = cv2.cvtColor(frame_or_path, cv2.COLOR_BGR2RGB)
            self.imagen_pil = Image.fromarray(frame_rgb)
        else:
            self.imagen_pil = frame_or_path.convert("RGB")

        self.ancho, self.alto = self.imagen_pil.size
        self.imagen = self.imagen_pil.resize(target_size, resampling)
        
        if show_image:
            temp_path = "debug_imagen.jpg"
            self.imagen.save(temp_path)
            import os
            os.system(f"explorer.exe {temp_path}")
    
    def make_image_boxes(self, prompt: str):
        """
        Pasa la imagen y el prompt (texto) a la IA para buscar múltiples objetos.
        Ejemplo: prompt="orange ball. robot. playing field."
        """
        import torch
        from contextlib import nullcontext
        
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
        inputs = self.processor(images=self.imagen, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            with autocast_ctx:
                outputs = self.model(**inputs)
                
        # Extracción y conversión de outputs (Diccionario o Dataclass)
        if isinstance(outputs, dict):
            self.masks = outputs.get("pred_masks", outputs.get("masks"))
            self.boxes = outputs.get("pred_boxes", outputs.get("boxes"))
            logits = outputs.get("pred_logits")
            self.scores = torch.sigmoid(logits) if logits is not None else None
        else:
            self.masks = getattr(outputs, 'pred_masks', getattr(outputs, 'masks', None))
            self.boxes = getattr(outputs, 'pred_boxes', getattr(outputs, 'boxes', None))
            logits = getattr(outputs, 'pred_logits', None)
            self.scores = torch.sigmoid(logits) if logits is not None else None
        
        return outputs

    def visualize_image_boxes(self):
        """
        Filtra los resultados por confianza y dibuja los rectángulos de los objetos detectados.
        """
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import time

        img_dibujada = self.imagen.copy()
        ancho_real, alto_real = img_dibujada.size
    
        # ==========================================
        # 1. DIBUJAR MÁSCARAS (Segmentación)
        # ==========================================
        if getattr(self, 'masks', None) is not None and len(self.masks) > 0:
            img_dibujada = img_dibujada.convert("RGBA")
            masks_np = (self.masks.cpu().to(torch.float32).numpy() * 255).astype(np.uint8)
            
            mascaras_2d = []
            for m in masks_np:
                m_sq = np.squeeze(m)
                if len(m_sq.shape) == 2:
                    mascaras_2d.append(m_sq)
                elif len(m_sq.shape) == 3:
                    mascaras_2d.append(m_sq[0])
                    
            n_masks = len(mascaras_2d)
            if n_masks > 0:
                cmap = plt.colormaps.get_cmap("rainbow").resampled(max(n_masks, 1))
                colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

                for mask_arr, color in zip(mascaras_2d, colors):
                    mask_img = Image.fromarray(mask_arr, mode="L") 
                    overlay = Image.new("RGBA", img_dibujada.size, color + (0,))
                    alpha = mask_img.point(lambda v: int(v * 0.5))
                    if alpha.size != overlay.size:
                        alpha = alpha.resize(overlay.size)
                    overlay.putalpha(alpha)
                    img_dibujada = Image.alpha_composite(img_dibujada, overlay)
                
        # ==========================================
        # 2. FILTRAR Y DIBUJAR CAJAS MULTI-OBJETO
        # ==========================================
        img_dibujada = img_dibujada.convert("RGB") 
        draw = ImageDraw.Draw(img_dibujada)
        
        if getattr(self, 'boxes', None) is not None and len(self.boxes) > 0 and self.boxes.shape[0] > 0:
            cajas_tensor = self.boxes[0] if len(self.boxes.shape) > 2 else self.boxes

            if getattr(self, 'scores', None) is None:
                num_cajas = cajas_tensor.shape[0]
                puntajes_tensor = torch.ones(num_cajas, device=getattr(self, 'device', 'cpu'))
            else:
                puntajes_tensor = self.scores[0] if len(self.scores.shape) > 1 else self.scores
            
            cajas = cajas_tensor.cpu().to(torch.float32).numpy()
            puntajes = puntajes_tensor.cpu().to(torch.float32).numpy()
            
            # Filtro de confianza configurable
            umbral = getattr(self, 'conf_threshold', 0.25)
            cajas_dibujadas = 0
            
            for caja, puntaje_raw in zip(cajas, puntajes):
                puntaje = np.max(puntaje_raw) if isinstance(puntaje_raw, np.ndarray) else puntaje_raw
                
                # Si no supera la confianza mínima, se ignora la caja
                if puntaje < umbral:
                    continue
                    
                cajas_dibujadas += 1
                
                # Escalado de coordenadas normalizadas (0 a 1) al tamaño real de la imagen
                if np.max(caja) <= 1.01:
                    x1 = int(caja[0] * ancho_real)
                    y1 = int(caja[1] * alto_real)
                    x2 = int(caja[2] * ancho_real)
                    y2 = int(caja[3] * alto_real)
                else:
                    x1, y1, x2, y2 = int(caja[0]), int(caja[1]), int(caja[2]), int(caja[3])

                caja_coordenadas = [x1, y1, x2, y2]
                
                # Dibujar recuadro e indicador de porcentaje
                draw.rectangle(caja_coordenadas, outline="#00FFFF", width=4) 
                
                texto = f"{puntaje * 100:.1f}%"
                y_text = max(0, y1 - 20)
                draw.rectangle([x1, y_text, x1 + 50, y_text + 20], fill="#00FFFF")
                draw.text((x1 + 2, y_text + 2), texto, fill="black")
                
            print(f"✅ ¡Procesamiento completo! Se detectaron y dibujaron {cajas_dibujadas} objetos válidos.")
                
        else:
            print("⚠️ No se encontraron cajas en la imagen.")
                
        # ==========================================
        # 3. GUARDADO DE LA IMAGEN
        # ==========================================
        nombre_unico = f"resultado_{int(time.time())}.jpg"
        img_dibujada.save(nombre_unico)
        print(f"📸 Archivo guardado como: '{nombre_unico}'")