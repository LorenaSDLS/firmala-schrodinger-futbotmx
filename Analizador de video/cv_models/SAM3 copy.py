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
import numpy as np

class SAM3():
    def __init__(self, api_path: str, ruta_dora: str = None, ruta_loha: str = None, mode: Literal["LoHa", "DoRa"] = "DoRa"):
        # --- 1. LIMPIEZA DE MEMORIA (Solo por si acaso) ---
        torch.cuda.empty_cache()
        gc.collect()

        # --- 2. INICIAR SESION ---
        with open(api_path, "r") as f:
            key = json.load(f)["API"]
        login(token=key)
        if mode not in ["LoHa", "DoRa"]:
            raise ValueError(f"Error fatal: El modo '{mode}' no es válido. Usa 'LoHa' o 'DoRa'.")
        self.mode = mode
        self.ruta_loha = ruta_loha
        self.ruta_dora = ruta_dora

        # --- 3. CARGAR EL MODELO ---
        self._load_model(self.mode)

    def change_mode(self, mode: Literal["LoHa", "DoRa"] = "DoRa"):
        if mode not in ["LoHa", "DoRa"]:
            raise ValueError(f"Error fatal: El modo '{mode}' no es válido. Usa 'LoHa' o 'DoRa'.")

        if self.mode == mode:
            print(f"SAM3 ya en el modo de entrenamiento {mode}")
        else:
            self._load_model(mode)
        

    def _load_model(self, mode: Literal["LoHa", "DoRa"] = "DoRa", conf_threshold: float = 0.50):

        torch.cuda.empty_cache()
        gc.collect()
        
        if mode not in ["LoHa", "DoRa"]:
            raise ValueError(f"Error fatal: El modo '{mode}' no es válido. Usa 'LoHa' o 'DoRa'.")

        if self.mode == "LoHa":
            self.ruta = self.ruta_loha
        elif self.mode == "DoRa":
            self.ruta = self.ruta_dora
        else:
            raise Exception("'LoHa' or 'DoRa' should be the input for mode...")

        if 'sam' not in globals():
            print("Cargando el modelo por primera vez... (esto tomará un momento)")
            self.device = "cpu"
            
            # 1. Guardamos el umbral directamente en tu clase, NO en el procesador
            self.conf_threshold = conf_threshold
            
            # 2. Le asignamos su nombre correcto: PROCESSOR
            self.processor = Sam3Processor.from_pretrained(self.ruta)
            
            # Nota: Al estar en CPU, float16 puede dar problemas o ser lentísimo.
            # Cambiamos temporalmente a float32 para asegurar estabilidad en CPU.
            self.base_model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float32)
            
            self.model = PeftModel.from_pretrained(self.base_model, self.ruta)
            self.model.to(self.device)
            self.model.eval()
            print(f"Modelo cargado exitosamente a {self.device}")
        else:
            print("El modelo ya está en la memoria. Saltando carga...")
            prev_conf = self.conf_threshold
            self.conf_threshold = conf_threshold
            print(f"Se cambió el confidence threshold a {conf_threshold}, valor previo {prev_conf}")



    # --- 4. HACER UN METODO PARA CARGAR LA IMAGEN ---
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
    
    # --- 5. HACER LA PREDICCION CON LA PROMPT ---
    def make_image_boxes(self, prompt: str):
        import torch
        from contextlib import nullcontext
        
        start_time = time.perf_counter()
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
        
        # 1. El procesador prepara la imagen y el texto (Esto ya lo tenías perfecto)
        inputs = self.processor(images=self.imagen, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            with autocast_ctx:
                # 2. Inferencia directa: Pasamos los inputs de golpe al modelo
                outputs = self.model(**inputs)
                
        # 3. Extracción de resultados estilo Hugging Face
        # Buscamos los atributos más comunes que usa HF (pred_boxes, boxes, iou_scores, etc.)
        end_time = time.perf_counter()

        print(f"Finished image analysis in {end_time - start_time}...")

        # --- NUEVO: ESCÁNER DE ENTRAÑAS ---
        print("\n" + "="*40)
        print("🕵️‍♂️ RADIOGRAFÍA DE LOS OUTPUTS DEL MODELO")
        print("="*40)
        if isinstance(outputs, dict):
            print("Formato: Diccionario")
            print("Variables encontradas:", list(outputs.keys()))
        else:
            print("Formato: Objeto (Dataclass)")
            variables = [a for a in dir(outputs) if not a.startswith('_')]
            print("Variables encontradas:", variables)
            
        print("="*40 + "\n")

        if isinstance(outputs, dict):
            # Por si el modelo devuelve un diccionario puro
            self.masks = outputs.get("pred_masks", outputs.get("masks"))
            self.boxes = outputs.get("pred_boxes", outputs.get("boxes"))
            logits = outputs.get("pred_logits")
            if logits is not None:
                # Convertimos los "logits" crudos a porcentajes de 0 a 1 usando Sigmoid
                self.scores = torch.sigmoid(logits)
            else:
                self.scores = None
        else:
            # Por si devuelve una clase tipo Dataclass (lo más común en HF)
            self.masks = getattr(outputs, 'pred_masks', getattr(outputs, 'masks', None))
            self.boxes = getattr(outputs, 'pred_boxes', getattr(outputs, 'boxes', None))
            
            # 👇 --- ESTA ES LA LÍNEA QUE TE FALTABA --- 👇
            logits = getattr(outputs, 'pred_logits', None)
            
            if logits is not None:
                # Convertimos los "logits" crudos a porcentajes de 0 a 1 usando Sigmoid
                self.scores = torch.sigmoid(logits)
            else:
                self.scores = None

        # 4. Consola de Diagnóstico
        print("\n--- DEBUG INFO ---")
        if self.boxes is not None:
            print(f"✅ ¡Éxito! Cajas encontradas (shape): {self.boxes.shape}")
            # Usamos hasattr por si los scores son un tensor con .shape o solo un número
            print(f"📊 Confianzas obtenidas: {self.scores.shape if hasattr(self.scores, 'shape') else self.scores}")
        else:
            print("❌ Boxes es None. El modelo procesó la imagen correctamente pero las cajas tienen otro nombre.")
            # Este print es magia: nos dirá exactamente qué variables generó el modelo
            nombres_disponibles = list(outputs.keys()) if isinstance(outputs, dict) else [a for a in dir(outputs) if not a.startswith('_')]
            print(f"🔎 Atributos disponibles dentro del output: {nombres_disponibles}")
            
        print("--------------------\n")
        
        return outputs

    def visualize_image_boxes(self):
        # 1. Imports hasta arriba para evitar errores de UnboundLocalError
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        import time

        img_dibujada = self.imagen.copy()
        ancho_real, alto_real = img_dibujada.size
    
        # ==========================================
        # 2. DIBUJAR MÁSCARAS (Si el modelo las generó)
        # ==========================================
        if getattr(self, 'masks', None) is not None and len(self.masks) > 0:
            img_dibujada = img_dibujada.convert("RGBA")
            
            # Convertir de bfloat16 a float32 antes de numpy()
            masks_np = (self.masks.cpu().to(torch.float32).numpy() * 255).astype(np.uint8)
            
            # Extraemos máscaras 2D limpias sin dimensiones extra
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
        # 3. DIBUJAR LA CAJA GANADORA (TOP-1)
        # ==========================================
        img_dibujada = img_dibujada.convert("RGB") 
        draw = ImageDraw.Draw(img_dibujada)
        
        if getattr(self, 'boxes', None) is not None and len(self.boxes) > 0 and self.boxes.shape[0] > 0:
            
            # Extraemos los tensores limpiando dimensiones fantasmas de Batch [1, 200, 4] -> [200, 4]
            cajas_tensor = self.boxes[0] if len(self.boxes.shape) > 2 else self.boxes

            if getattr(self, 'scores', None) is None:
                # Si por alguna razón extrema no hay puntajes, creamos unos ficticios
                num_cajas = cajas_tensor.shape[0]
                puntajes_tensor = torch.ones(num_cajas, device=getattr(self, 'device', 'cpu'))
            else:
                puntajes_tensor = self.scores[0] if len(self.scores.shape) > 1 else self.scores
            
            # Convertir a numpy seguro en float32
            cajas = cajas_tensor.cpu().to(torch.float32).numpy()
            puntajes = puntajes_tensor.cpu().to(torch.float32).numpy()
            
            # 🥇 ¡EL TRUCO PARA UNA SOLA CAJA (TOP-1)! 🥇
            mejor_indice = np.argmax(puntajes) # Encuentra el número de la caja con mayor confianza
            mejor_caja = cajas[mejor_indice]
            mejor_puntaje = puntajes[mejor_indice]
            
            # Extraemos el valor real por si el puntaje es un array de 1 elemento
            mejor_puntaje = np.max(mejor_puntaje) if isinstance(mejor_puntaje, np.ndarray) else mejor_puntaje
            
            print(f"🏆 La mejor detección tiene una confianza de: {mejor_puntaje * 100:.1f}%")
            
            # Usamos el umbral configurado o 0.20 por defecto (20% es suficiente si es la mejor absoluta)
            umbral = getattr(self, 'conf_threshold', 0.20) 
            
            if mejor_puntaje >= umbral:
                # IMPORTANTE: Escalar coordenadas si están normalizadas (entre 0 y 1)
                if np.max(mejor_caja) <= 1.01:
                    x1 = int(mejor_caja[0] * ancho_real)
                    y1 = int(mejor_caja[1] * alto_real)
                    x2 = int(mejor_caja[2] * ancho_real)
                    y2 = int(mejor_caja[3] * alto_real)
                else:
                    x1, y1, x2, y2 = int(mejor_caja[0]), int(mejor_caja[1]), int(mejor_caja[2]), int(mejor_caja[3])

                caja_coordenadas = [x1, y1, x2, y2]
                
                # Dibujar rectángulo verde fluorescente
                draw.rectangle(caja_coordenadas, outline="#00FF00", width=4) 
                
                # Fondo para el texto
                texto = f"Pelota: {mejor_puntaje:.2f}"
                y_text = max(0, y1 - 20)
                draw.rectangle([x1, y_text, x1 + 80, y_text + 20], fill="#00FF00")
                
                # Texto negro
                draw.text((x1 + 2, y_text + 2), texto, fill="black")
                print("✅ ¡Se dibujó únicamente la pelota ganadora!")
            else:
                print(f"⚠️ Ninguna caja superó el umbral mínimo del {umbral*100}%.")
                
        else:
            print("⚠️ No hay cajas detectadas por el modelo.")
                
        # ==========================================
        # 4. GUARDADO DE LA IMAGEN FINAL
        # ==========================================
        # Generamos un nombre único usando los segundos del reloj para que VS Code no nos engañe
        nombre_unico = f"resultado_{int(time.time())}.jpg"
        
        img_dibujada.save(nombre_unico)
        
        print(f"📸 ¡Éxito! Imagen real guardada como '{nombre_unico}'.")
        print(f"👆 Busca EXACTAMENTE ese archivo en el explorador de VS Code y ábrelo.")