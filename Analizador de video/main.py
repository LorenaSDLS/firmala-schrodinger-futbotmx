from tracker.tracker import FutBotTracker

def probar_sistema():
    print("=====================================================")
    print("   ⚽ BIENVENIDO AL SISTEMA DE TRACKING FUTBOTMX ⚽")
    print("=====================================================\n")

    # Aquí inicializamos la clase. Solo se hace una vez y carga los modelos pesados.
    print("Cargando el cerebro del sistema...")
    mi_tracker = FutBotTracker(
        ruta_yolo="YOLO/best.pt",
        ruta_sam_api="api_sam3.json", ### LA API TIENE QUE SER SU LLAVE PARA HUGGINGFACE
        mode="LoHa"
    )

    # Las rutas de los archivos que vamos a probar
    imagen_prueba = "debug_imagen.jpg"
    video_prueba = "video-893_singular_display.mov"


    # HERRAMIENTA 1: EL CALIBRADOR INTERACTIVO 
    # ¿Que hace?: Abre la foto. Si haces clic en la cancha, imprime las coordenadas (X, Y) en la consola.
    # ¿Para qué sirve?: Para sacar los puntos exactos y armar la Matriz de Homografía super rapido. (Homography Matrix)
    # ¿Qué regresa?: Nada en codigo, pero imprime los pixeles en la terminal. ;)
    
    # print("\n--- INICIANDO PRUEBA 1: Calibrador ---")
    # mi_tracker.calibrador_interactivo(imagen_prueba)


    # HERRAMIENTA 2: DEPURACIÓN RÁPIDA (UNA SOLA FOTO)
    # ¿Que hace?: Fuerza a SAM 3 y YOLO a escanear una sola foto y dibuja los resultados.
    # ¿Para qué sirve?: Para ver qué está detectando la IA sin tener que esperar a que corra todo un video.
    # ¿Qué regresa?: Una ventana estática con las cajas dibujadas. Se cierra presionando cualquier tecla.
    
    # print("\n--- INICIANDO PRUEBA 2: Análisis de Fotograma Estático ---")
    # mi_tracker.procesar_imagen_debug(imagen_prueba)


    # HERRAMIENTA 3: EL TRACKING COMPLETO
    # ¿Que hace?: Corre el algoritmo completo híbrido (YOLO + SAM3 + logica anti-oclusión).
    # ¿Para que sirve?: Es el producto final. Analiza el partido en tiempo real. (Tarda un poquillo, sobre todo si la pelota se pierde)
    # ¿Que regresa?: El reproductor de video en vivo con el HUD, cajas y control inteligente. Se cierra presionando 'q'.
    
    print("\n--- INICIANDO PRUEBA 3: Procesamiento de Video Híbrido ---")
    mi_tracker.procesar_video(video_prueba, guardar_como="partido_analizado.mp4")


if __name__ == "__main__":
    probar_sistema()