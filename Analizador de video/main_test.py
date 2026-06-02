from cv_models.SAM3 import SAM3
from cv_models.YOLO import YouOnlyLiveOnce
from tracker.tracker import FutBo
# Asegúrate de que esta línea coincida con dónde tienes guardada tu clase YOLO:
# from cv_models.YOLO import YouOnlyLiveOnce 
from random import choice, seed
from pathlib import Path
import cv2

print("Iniciando sistemas...")
yolo = YouOnlyLiveOnce(yolo_pt_path="/mnt/d/Documentos/FutBot/firmala-schrodinger-futbotmx/runs/detect/train/weights/best.pt")
sam = SAM3("api_sam3.json", 
               ruta_loha="resultados_sam3_LoHa_final", 
               ruta_dora="Resultados_DoRa",
               mode="LoHa")
sam._load_model(mode="DoRa", conf_threshold=0.40)

cap = cv2.VideoCapture("video-893_singular_display.mov")
print("Video cargado exitosamente!")


# =====================================================================
#            INICIALIZACIÓN DE ESTRUCTURAS Y TOLERANCIAS
# =====================================================================
estados = {
    "campo": None,
    "pelota": None,
    "robot_0": None,
    "robot_1": None,
    "robot_2": None,
    "robot_3": None
}

frames_perdidos = {
    "campo": 0,
    "pelota": 0,
    "robot_0": 0,
    "robot_1": 0,
    "robot_2": 0,
    "robot_3": 0
}

# Límites de paciencia (en frames) antes de mandar a llamar a SAM 3 (30 fps, entonces para tener segundos TOL / 30 = s)
TOLERANCIA_PELOTA_NORMAL = 15  
TOLERANCIA_PELOTA_OCULTA = 150
TOLERANCIA_ROBOTS_NORMAL = 30
TOLERANCIA_ROBOTS_OCULTO = 90
TOLERANCIA_CAMPO = 60

numero_frame = 0
sam3_asesoramiento_inicial = False

# MEMORIA FOTOGRÁFICA
ultima_posicion_pelota = None 
ultimas_posiciones_robots = {
    "robot_0": None, "robot_1": None, 
    "robot_2": None, "robot_3": None
}

# =====================================================================
#                 FUNCIONES DE RENDERIZADO Y CONTROL
# =====================================================================
def dibujar_hud_estado(frame, yolo_activo=True, sam3_activo=False):
    """
    Dibuja un HUD semitransparente que muestra el estado de ambos modelos en tiempo real.
    """
    # Fondo negro semitransparente con borde gris
    cv2.rectangle(frame, (10, 10), (260, 85), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (260, 85), (100, 100, 100), 1)

    # SECCIÓN YOLOv8
    if yolo_activo:
        cv2.putText(frame, "YOLOv8: TRACKING", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "YOLOv8: INACTIVO", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # SECCIÓN SAM 3
    if sam3_activo:
        cv2.putText(frame, "SAM 3:  AL RESCATE", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "SAM 3:  EN ESPERA", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)


def ejecutar_rescate_sam3(frame_actual, buscar_campo=True, buscar_pelota=True, buscar_robots=True):
    """
    SAM 3 inteligente con rescate dirigido.
    Solución definitiva y blindada contra cualquier cambio de dimensiones en PyTorch.
    """
    print(f"🤖 SAM 3: Escaneando frame {numero_frame} (Rescate Dirigido)...")
    sam.load_image(frame_actual)

    # 1. Buscar Campo
    if buscar_campo:
        print("   -> SAM 3: Buscando el campo...")
        campo = sam.make_image_boxes(prompt="playing field")
        if campo and len(campo) > 0:
            print("   -> SAM 3: Encontró el campo!")
            caja_tensor = campo[0]
            
            if hasattr(caja_tensor, "detach"):
                box = caja_tensor.detach().cpu().view(-1).tolist()[:4]
            else:
                box = caja_tensor[:4] # Por si ya fuera lista
            
            estados["campo"] = [int(box[0]*sam.ancho), int(box[1]*sam.alto), int(box[2]*sam.ancho), int(box[3]*sam.alto)]
            frames_perdidos["campo"] = 0
        else:
            estados["campo"] = None
            print("   -> SAM 3: No encontró el campo")

    # 2. Buscar Pelota
    if buscar_pelota:
        print("   -> SAM 3: Buscando la pelota...")
        pelota = sam.make_image_boxes(prompt="orange ball")
        if pelota and len(pelota) > 0:
            print("   -> SAM 3: Encontró la pelota!")
            caja_tensor = pelota[0]
            
            # Aplicamos el mismo blindaje para la pelota
            if hasattr(caja_tensor, "detach"):
                box = caja_tensor.detach().cpu().view(-1).tolist()[:4]
            else:
                box = caja_tensor[:4]
            
            estados["pelota"] = [int(box[0]*sam.ancho), int(box[1]*sam.alto), int(box[2]*sam.ancho), int(box[3]*sam.alto)]
            frames_perdidos["pelota"] = 0
        else:
            estados["pelota"] = None
            print("   -> SAM 3: No encontró la pelota")

    # 3. Buscar Robots
    if buscar_robots:
        print("   -> SAM 3: Buscando los robots...")
        for i in range(4):
            estados[f"robot_{i}"] = None
            frames_perdidos[f"robot_{i}"] = 0

        robots = sam.make_image_boxes(prompt="robots")
        if robots and len(robots) > 0:
            print("   -> SAM 3: Encontró los robots!")
            robots_tensor = robots[0]
            
            # Convertimos la matriz completa [M, 4] en una lista de listas nativa de Python [[x1,y1,x2,y2], ...]
            if hasattr(robots_tensor, "detach"):
                robots_list = robots_tensor.detach().cpu().view(-1, 4).tolist()
            else:
                robots_list = robots
            
            # Ahora iteramos de forma 100% segura sobre las sublistas (máximo 4 robots)
            for i, box in enumerate(robots_list[:4]):
                x1 = int(box[0] * sam.ancho)
                y1 = int(box[1] * sam.alto)
                x2 = int(box[2] * sam.ancho)
                y2 = int(box[3] * sam.alto)
                estados[f"robot_{i}"] = [x1, y1, x2, y2]

        else:
            print("   -> SAM 3: No encontró los robots")


def dibujar_cajas_del_estado(frame_actual):
    """
    Lee el diccionario global 'estados' y dibuja las cajas en el frame.
    """
    # Dibujar Campo (Verde)
    if estados["campo"]:
        cx1, cy1, cx2, cy2 = estados["campo"]
        cv2.rectangle(frame_actual, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        cv2.putText(frame_actual, "Campo", (cx1, max(20, cy1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibujar Pelota (Naranja)
    if estados["pelota"]:
        px1, py1, px2, py2 = estados["pelota"]
        cv2.rectangle(frame_actual, (px1, py1), (px2, py2), (0, 165, 255), 2)
        cv2.putText(frame_actual, "Balon", (px1, max(20, py1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # Dibujar Robots (Azul)
    for i in range(4):
        if estados[f"robot_{i}"]:
            rx1, ry1, rx2, ry2 = estados[f"robot_{i}"]
            cv2.rectangle(frame_actual, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(frame_actual, f"Robot_{i}", (rx1, max(20, ry1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# =====================================================================
# BUCLE PRINCIPAL DE PROCESAMIENTO
# =====================================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    numero_frame += 1
    # 👇 OBTENEMOS LAS MEDIDAS DEL FRAME PARA SABER DÓNDE ESTÁN LOS BORDES 👇
    alto_frame, ancho_frame = frame.shape[:2]

    # -----------------------------------------------------------------
    # FASE 1: INICIALIZACIÓN MÁGICA CON SAM 3 (Solo corre en el Frame 1)
    # -----------------------------------------------------------------
    if not sam3_asesoramiento_inicial:
        print(f"\n🚀 Iniciando sistema. Buscando objetos con SAM 3 en Frame {numero_frame}...")
        
        ejecutar_rescate_sam3(frame, buscar_pelota=True, buscar_campo=True, buscar_robots=True)
        
        dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
        dibujar_cajas_del_estado(frame)
        
        print(f"✅ Inicialización terminada en píxeles reales. Control cedido a YOLO.\n")
        sam3_asesoramiento_inicial = True
        
        cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
        cv2.waitKey(1)
        continue

    # -----------------------------------------------------------------
    # FASE 2: DETECCIÓN ESTÁNDAR CON YOLOv8 (Frame 2 en adelante)
    # -----------------------------------------------------------------
    resultados_yolo = yolo.analyze_frame(frame)

    # Filtrado por clases
    robots_yolo = [obj["box"] for obj in resultados_yolo if obj["class_name"] == "robot"]
    pelotas_yolo = [obj["box"] for obj in resultados_yolo if obj["class_name"] == "orange ball"]
    campo_yolo = [obj["box"] for obj in resultados_yolo if obj["class_name"] == "field"]

    # --- Evaluación del Campo ---
    if len(campo_yolo) > 0:
        estados["campo"] = campo_yolo[0]
        frames_perdidos["campo"] = 0
    else:
        estados["campo"] = None
        frames_perdidos["campo"] += 1

    # --- Evaluación de la Pelota ---
    if len(pelotas_yolo) > 0:
        estados["pelota"] = pelotas_yolo[0]
        ultima_posicion_pelota = pelotas_yolo[0] # Guardamos la última posición conocida para optimizar el script
        frames_perdidos["pelota"] = 0
    else:
        estados["pelota"] = None 
        frames_perdidos["pelota"] += 1

    # --- Evaluación de los Robots ---
    for i in range(4):
        estados[f"robot_{i}"] = None

    for i, caja in enumerate(robots_yolo[:4]):
        estados[f"robot_{i}"] = caja
        ultimas_posiciones_robots[f"robot_{i}"] = caja # Guardamos la última posición conocida para optimizar el script

    for i in range(4):
        if estados[f"robot_{i}"] is not None:
            frames_perdidos[f"robot_{i}"] = 0
        else:
            frames_perdidos[f"robot_{i}"] += 1

    # -------------------------------------------------------------------------------------------
    # FASE 3: TOMA DE DECISIONES DE PERSISTENCIA (Lógica Anti-Borrachos)
    # -------------------------------------------------------------------------------------------
    
    # 0. 🚧 NUEVO: DETECCIÓN DE BORDES DE LA CÁMARA 🚧
    pelota_salio_de_pantalla = False
    MARGEN_BORDE = 50 # Si la pelota está a menos de 50 píxeles del borde, asumimos que salió
    
    if ultima_posicion_pelota is not None:
        px1, py1, px2, py2 = ultima_posicion_pelota
        # Si tocó el borde izquierdo, superior, derecho o inferior
        if px1 < MARGEN_BORDE or py1 < MARGEN_BORDE or px2 > (ancho_frame - MARGEN_BORDE) or py2 > (alto_frame - MARGEN_BORDE):
            pelota_salio_de_pantalla = True

    # 1. EVALUAR OCLUSIÓN DE LA PELOTA (Tu código intacto)
    tolerancia_pelota_actual = TOLERANCIA_PELOTA_NORMAL
    if frames_perdidos["pelota"] > 0 and ultima_posicion_pelota is not None:
        px1, py1, px2, py2 = ultima_posicion_pelota
        for i in range(4):
            caja_robot = estados[f"robot_{i}"]
            if caja_robot:
                rx1, ry1, rx2, ry2 = caja_robot
                m = 40
                if not (px2 < rx1-m or px1 > rx2+m or py2 < ry1-m or py1 > ry2+m):
                    tolerancia_pelota_actual = TOLERANCIA_PELOTA_OCULTA
                    break 

    # 2. EVALUAR OCLUSIÓN DE LOS ROBOTS (Tu código intacto)
    alerta_robots = False 
    for i in range(4):
        # ... (Tu código de robots que ya tenías) ...
        if frames_perdidos[f"robot_{i}"] > 0 and ultimas_posiciones_robots[f"robot_{i}"] is not None:
            rx1, ry1, rx2, ry2 = ultimas_posiciones_robots[f"robot_{i}"]
            tolerancia_este_robot = TOLERANCIA_ROBOTS_NORMAL
            for j in range(4):
                if i != j and estados[f"robot_{j}"] is not None:
                    ox1, oy1, ox2, oy2 = estados[f"robot_{j}"]
                    m = 40
                    if not (rx2 < ox1-m or rx1 > ox2+m or ry2 < oy1-m or ry1 > oy2+m):
                        tolerancia_este_robot = TOLERANCIA_ROBOTS_OCULTO
                        break
            if frames_perdidos[f"robot_{i}"] > tolerancia_este_robot:
                alerta_robots = True
                break

    # 3. --- DECISIONES FINALES (CON SENTIDO COMÚN) ---
    if frames_perdidos["pelota"] > tolerancia_pelota_actual:
        # 🚫 Filtro 1: No hay cancha
        if estados["campo"] is None:
            print(f"🙈 [Frame {numero_frame}] Cámara perdida. Esperando a ver la cancha...")
            dibujar_hud_estado(frame, yolo_activo=True, sam3_activo=False)
            
        # 🚫 Filtro 2: Salió de la pantalla
        elif pelota_salio_de_pantalla:
            print(f"➡️ [Frame {numero_frame}] Balón fuera de la pantalla. Esperando a que regrese...")
            dibujar_hud_estado(frame, yolo_activo=True, sam3_activo=False)
            
        # 🚨 Emergencia real: ¡Desapareció mágicamente en medio de la cancha!
        else:
            print(f"🚨 ALERTA: Balón perdido en medio del campo. SAM 3 al rescate...")
            dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
            cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
            cv2.waitKey(1) 
            ejecutar_rescate_sam3(frame, buscar_pelota=True, buscar_campo=False, buscar_robots=False)

    elif alerta_robots:
        # Aquí también podrías agregar validaciones si el robot salió de pantalla, pero con la pelota basta por ahora.
        print(f"🚨 ALERTA: Robot(s) perdidos superaron tolerancia de oclusión. SAM 3 al rescate...")
        dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
        cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
        cv2.waitKey(1)
        ejecutar_rescate_sam3(frame, buscar_pelota=False, buscar_campo=False, buscar_robots=True)

    elif frames_perdidos["campo"] > TOLERANCIA_CAMPO:
        print(f"🚨 ALERTA: Estructura de campo perdida. SAM 3 al rescate...")
        dibujar_hud_estado(frame, yolo_activo=False, sam3_activo=True)
        cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
        cv2.waitKey(1)
        ejecutar_rescate_sam3(frame, buscar_pelota=False, buscar_campo=True, buscar_robots=False)

    else:
        dibujar_hud_estado(frame, yolo_activo=True, sam3_activo=False)

    dibujar_cajas_del_estado(frame)

    # Mostrar frame renderizado
    cv2.imshow("Copa FutBotMX - Tracking Maestro", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()