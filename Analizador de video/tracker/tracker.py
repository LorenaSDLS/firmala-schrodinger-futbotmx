from cv_models.SAM3 import SAM3
from cv_models.YOLO import YouOnlyLiveOnce
import cv2
from typing import Literal
import numpy as np

class FutBotTracker:
    def __init__(self, ruta_yolo, ruta_sam_api, mode = Literal["DoRa", "LoHa"]):
        """
        Inicializa los modelos pesados de IA. Esto toma tiempo, así que 
        solo se hace una vez cuando se crea el objeto.
        """
        print("Iniciando sistemas de IA...")
        self.mode = mode
        self.yolo = YouOnlyLiveOnce(yolo_pt_path=ruta_yolo)
        self.sam = SAM3(ruta_sam_api, 
                        ruta_loha="LoHa", 
                        ruta_dora="DoRa",
                        mode=self.mode)
        
        # Tolerancias configurables (Paciencia en frames)
        self.TOL_PELOTA_NORMAL = 15  
        self.TOL_PELOTA_OCULTA = 150
        self.TOL_ROBOTS_NORMAL = 30
        self.TOL_ROBOTS_OCULTO = 90
        self.TOL_CAMPO = 60

        self._resetear_memoria()
        print("✅ Sistemas listos y cargados exitosamente.")

    def _resetear_memoria(self):
        """Limpia los estados para poder analizar un video nuevo desde cero."""
        self.estados = {"campo": None, "pelota": None, "robot_0": None, "robot_1": None, "robot_2": None, "robot_3": None}
        self.frames_perdidos = {"campo": 0, "pelota": 0, "robot_0": 0, "robot_1": 0, "robot_2": 0, "robot_3": 0}
        self.ultima_posicion_pelota = None 
        self.ultimas_posiciones_robots = {"robot_0": None, "robot_1": None, "robot_2": None, "robot_3": None}
        self.numero_frame = 0
        self.sam3_asesoramiento_inicial = False

    def _dibujar_hud_estado(self, frame, yolo_activo=True, sam3_activo=False):
        cv2.rectangle(frame, (10, 10), (260, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (260, 85), (100, 100, 100), 1)
        if yolo_activo:
            cv2.putText(frame, "YOLOv8: TRACKING", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "YOLOv8: INACTIVO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if sam3_activo:
            cv2.putText(frame, "SAM 3:  AL RESCATE", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "SAM 3:  EN ESPERA", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    def _dibujar_cajas(self, frame):
        if self.estados["campo"]:
            cx1, cy1, cx2, cy2 = self.estados["campo"]
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.putText(frame, "Campo", (cx1, max(20, cy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.estados["pelota"]:
            px1, py1, px2, py2 = self.estados["pelota"]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
            cv2.putText(frame, "Balon", (px1, max(20, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        for i in range(4):
            if self.estados[f"robot_{i}"]:
                rx1, ry1, rx2, ry2 = self.estados[f"robot_{i}"]
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                cv2.putText(frame, f"Robot_{i}", (rx1, max(20, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _ejecutar_rescate_sam3(self, frame_actual, buscar_campo=True, buscar_pelota=True, buscar_robots=True):

        print(f"SAM 3: Escaneando frame {self.numero_frame} (Rescate Dirigido)...")
        self.sam.load_image(frame_actual)

        # Obtenemos la confianza de SAM (el 0.25 que pusiste)
        umbral = getattr(self.sam, 'conf_threshold', 0.25)

        def obtener_mejores_cajas():
            """Filtra la basura de la IA y nos devuelve las coordenadas reales de las cajas."""
            if getattr(self.sam, 'boxes', None) is None or getattr(self.sam, 'scores', None) is None:
                return []
                
            cajas_tensor = self.sam.boxes[0] if len(self.sam.boxes.shape) > 2 else self.sam.boxes
            puntajes_tensor = self.sam.scores[0] if len(self.sam.scores.shape) > 1 else self.sam.scores
            
            cajas = cajas_tensor.detach().cpu().numpy()
            puntajes = puntajes_tensor.detach().cpu().numpy()
            
            cajas_validas = []
            for caja, puntaje_raw in zip(cajas, puntajes):
                p = np.max(puntaje_raw)
                if p >= umbral: 
                    # Escalamos las coordenadas al tamaño real de la imagen
                    if np.max(caja) <= 1.01:
                        x1 = int(caja[0] * self.sam.ancho)
                        y1 = int(caja[1] * self.sam.alto)
                        x2 = int(caja[2] * self.sam.ancho)
                        y2 = int(caja[3] * self.sam.alto)
                    else:
                        x1, y1, x2, y2 = int(caja[0]), int(caja[1]), int(caja[2]), int(caja[3])
                    cajas_validas.append((p, [x1, y1, x2, y2]))
            
            # Ordenamos de mayor a menor confianza
            cajas_validas.sort(key=lambda x: x[0], reverse=True)
            return [c[1] for c in cajas_validas]

        # --- 1. BUSCAR CAMPO ---
        if buscar_campo:
            print("   -> SAM 3: Buscando el campo...")
            self.sam.make_image_boxes(prompt="playing field")
            cajas_encontradas = obtener_mejores_cajas()
            if cajas_encontradas:
                self.estados["campo"] = cajas_encontradas[0]
                self.frames_perdidos["campo"] = 0
                print("   -> SAM 3: ¡Encontró el campo!")
            else:
                self.estados["campo"] = None
                print("   -> SAM 3: No encontró el campo")

        # --- 2. BUSCAR PELOTA ---
        if buscar_pelota:
            print("   -> SAM 3: Buscando la pelota...")
            self.sam.make_image_boxes(prompt="orange ball")
            cajas_encontradas = obtener_mejores_cajas()
            if cajas_encontradas:
                self.estados["pelota"] = cajas_encontradas[0]
                self.frames_perdidos["pelota"] = 0
                print("   -> SAM 3: ¡Encontró la pelota!")
            else:
                self.estados["pelota"] = None
                print("   -> SAM 3: No encontró la pelota")

        # --- 3. BUSCAR ROBOTS ---
        if buscar_robots:
            print("   -> SAM 3: Buscando los robots...")
            for i in range(4):
                self.estados[f"robot_{i}"] = None
                self.frames_perdidos[f"robot_{i}"] = 0

            self.sam.make_image_boxes(prompt="robots")
            cajas_encontradas = obtener_mejores_cajas()
            if cajas_encontradas:
                for i, box in enumerate(cajas_encontradas[:4]):
                    self.estados[f"robot_{i}"] = box
                print(f"   -> SAM 3: ¡Encontró {min(len(cajas_encontradas), 4)} robots!")
            else:
                print("   -> SAM 3: No encontró los robots")



    def _dibujar_puntos_homografia(self, frame, puntos_pixeles):
        """Dibuja círculos llamativos en los puntos que se usarán para la matriz de homografía."""
        for (x, y) in puntos_pixeles:
            cv2.circle(frame, (int(x), int(y)), radius=8, color=(255, 0, 255), thickness=-1)
            cv2.circle(frame, (int(x), int(y)), radius=12, color=(0, 255, 255), thickness=2)


    def calibrador_interactivo(self, ruta_imagen):
        """
        Abre una imagen estática. Si el usuario hace clic en la pantalla,
        imprime las coordenadas (X, Y) exactas en la terminal.
        """
        frame = cv2.imread(ruta_imagen)
        if frame is None:
            print(f"❌ Error: No se pudo abrir la imagen {ruta_imagen}")
            return

        print("\n" + "="*50)
        print(" MODO CALIBRACIÓN ACTIVADO ")
        print("Haz clic en cualquier parte de la ventana para obtener la coordenada.")
        print("Presiona 'q' para salir.")
        print("="*50 + "\n")

        def evento_click(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"📍 Punto capturado -> X: {x:4d} | Y: {y:4d}")
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Calibrador de Cancha", frame)

        cv2.imshow("Calibrador de Cancha", frame)
        cv2.setMouseCallback("Calibrador de Cancha", evento_click)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def procesar_imagen_debug(self, ruta_imagen, ruta_resultado = None):
        """
        Procesa UNA SOLA imagen en lugar de un video.
        Guarda el resultado en disco para evadir los bugs visuales de WSL.
        """
        frame = cv2.imread(ruta_imagen)
        if frame is None:
            print(f"❌ Error: No se pudo cargar la imagen {ruta_imagen}")
            return

        print(f"🔍 Analizando imagen de debug: {ruta_imagen}")
        self._resetear_memoria()
        
        self._ejecutar_rescate_sam3(frame, buscar_campo=True, buscar_pelota=True, buscar_robots=True)
        
        self._dibujar_cajas(frame)
        self._dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)

        if not ruta_resultado:
            ruta_salida = "resultado_debug.jpg"
        else:
            ruta_salida = ruta_resultado
        cv2.imwrite(ruta_salida, frame)
        print(f"Analisis terminado. Imagen guardada exitosamente como: '{ruta_salida}'")

        cv2.namedWindow("Modo Debug - Fotograma Estático", cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow("Modo Debug - Fotograma Estático", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): 
                break
                
        cv2.destroyAllWindows()

    def procesar_video(self, ruta_video, guardar_como=None):
        """
        Función principal de tracking. Analiza un video completo frame por frame.
        Si le pasas 'guardar_como="resultado.mp4"', guardará el video procesado.
        """
        self._resetear_memoria() 
        cap = cv2.VideoCapture(ruta_video)
        
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir el video {ruta_video}")
            return

        print(f"🎬 Iniciando analisis de: {ruta_video}")

        out = None
        if guardar_como:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30
            
            ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(guardar_como, fourcc, fps, (ancho, alto))
            print(f"💾 El video final se guardará como: {guardar_como}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            self.numero_frame += 1
            alto_frame, ancho_frame = frame.shape[:2]

            # INICIALIZACIÓN
            if not self.sam3_asesoramiento_inicial:
                self._ejecutar_rescate_sam3(frame, buscar_pelota=True, buscar_campo=True, buscar_robots=True)
                self._dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
                self._dibujar_cajas(frame)
                self.sam3_asesoramiento_inicial = True
                
                if out: out.write(frame)
                
                cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
                cv2.waitKey(1)
                continue

            # YOLO
            resultados_yolo = self.yolo.analyze_frame(frame)
            robots_yolo = [obj["box"] for obj in resultados_yolo if obj["class_name"] == "robot"]
            pelotas_yolo = [obj["box"] for obj in resultados_yolo if obj["class_name"] == "orange ball"]
            campo_yolo = [obj["box"] for obj in resultados_yolo if obj["class_name"] == "field"]

            if campo_yolo:
                self.estados["campo"] = campo_yolo[0]
                self.frames_perdidos["campo"] = 0
            else:
                self.estados["campo"] = None
                self.frames_perdidos["campo"] += 1

            if pelotas_yolo:
                self.estados["pelota"] = pelotas_yolo[0]
                self.ultima_posicion_pelota = pelotas_yolo[0]
                self.frames_perdidos["pelota"] = 0
            else:
                self.estados["pelota"] = None 
                self.frames_perdidos["pelota"] += 1

            for i in range(4): self.estados[f"robot_{i}"] = None
            for i, caja in enumerate(robots_yolo[:4]):
                self.estados[f"robot_{i}"] = caja
                self.ultimas_posiciones_robots[f"robot_{i}"] = caja

            for i in range(4):
                if self.estados[f"robot_{i}"] is not None: self.frames_perdidos[f"robot_{i}"] = 0
                else: self.frames_perdidos[f"robot_{i}"] += 1

            # LOGICA ANTI-BORRACHOS Y OCLUSION
            pelota_fuera = False
            M_BORDE = 50
            if self.ultima_posicion_pelota:
                px1, py1, px2, py2 = self.ultima_posicion_pelota
                if px1 < M_BORDE or py1 < M_BORDE or px2 > (ancho_frame - M_BORDE) or py2 > (alto_frame - M_BORDE):
                    pelota_fuera = True

            tol_pelota = self.TOL_PELOTA_NORMAL
            if self.frames_perdidos["pelota"] > 0 and self.ultima_posicion_pelota:
                px1, py1, px2, py2 = self.ultima_posicion_pelota
                for i in range(4):
                    if self.estados[f"robot_{i}"]:
                        rx1, ry1, rx2, ry2 = self.estados[f"robot_{i}"]
                        if not (px2 < rx1-40 or px1 > rx2+40 or py2 < ry1-40 or py1 > ry2+40):
                            tol_pelota = self.TOL_PELOTA_OCULTA
                            break 

            alerta_robots = False 
            for i in range(4):
                if self.frames_perdidos[f"robot_{i}"] > 0 and self.ultimas_posiciones_robots[f"robot_{i}"]:
                    rx1, ry1, rx2, ry2 = self.ultimas_posiciones_robots[f"robot_{i}"]
                    tol_robot = self.TOL_ROBOTS_NORMAL
                    for j in range(4):
                        if i != j and self.estados[f"robot_{j}"]:
                            ox1, oy1, ox2, oy2 = self.estados[f"robot_{j}"]
                            if not (rx2 < ox1-40 or rx1 > ox2+40 or ry2 < oy1-40 or ry1 > oy2+40):
                                tol_robot = self.TOL_ROBOTS_OCULTO
                                break
                    if self.frames_perdidos[f"robot_{i}"] > tol_robot:
                        alerta_robots = True
                        break

            # DECISIONES FINALES
            if self.frames_perdidos["pelota"] > tol_pelota:
                if not self.estados["campo"] or pelota_fuera:
                    self._dibujar_hud_estado(frame, yolo_activo=True, sam3_activo=False)
                else:
                    self._dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
                    cv2.imshow("Copa FutBotMX - Tracking Maestro", frame); cv2.waitKey(1) 
                    self._ejecutar_rescate_sam3(frame, buscar_pelota=True, buscar_campo=False, buscar_robots=False)
            elif alerta_robots:
                self._dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
                cv2.imshow("Copa FutBotMX - Tracking Maestro", frame); cv2.waitKey(1)
                self._ejecutar_rescate_sam3(frame, buscar_pelota=False, buscar_campo=False, buscar_robots=True)
            elif self.frames_perdidos["campo"] > self.TOL_CAMPO:
                self._dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
                cv2.imshow("Copa FutBotMX - Tracking Maestro", frame); cv2.waitKey(1)
                self._ejecutar_rescate_sam3(frame, buscar_pelota=False, buscar_campo=True, buscar_robots=False)
            else:
                self._dibujar_hud_estado(frame, yolo_activo=True, sam3_activo=False)

            # Dibujamos las cajas finales
            self._dibujar_cajas(frame)

            if out:
                out.write(frame)

            cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        if out:
            out.release()
            print(f"✅ Video guardado con éxito en: {guardar_como}")
            
        cv2.destroyAllWindows()