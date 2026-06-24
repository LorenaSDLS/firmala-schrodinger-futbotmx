# Comparación del pipeline de tracking: V1 vs V8

## 1. Objetivo del documento

Este documento resume los cambios realizados entre la primera versión del pipeline de tracking (**V1**) y la versión más reciente analizada (**V8**). La comparación se hizo directamente sobre ambos repositorios, considerando la arquitectura, detección, tracking, identidad de robots, geometría de cancha, eventos, exportación, simulación, depuración y productos finales.

El objetivo no es describir cada modificación línea por línea, sino explicar de manera clara qué capacidades se conservaron, cuáles se rediseñaron y qué problemas se resolvieron durante la evolución del sistema.

---

## 2. Resumen ejecutivo

La V1 ya era un pipeline funcional: cargaba y validaba el video, ejecutaba detección con YOLO, mantenía tracks temporales de robots y pelota, detectaba eventos básicos y generaba archivos para una repetición esquemática. Sin embargo, las posiciones seguían dependiendo principalmente de las coordenadas de imagen o de una caja rectangular aproximada de la cancha. Tampoco existía una separación robusta entre identidad temporal, identidad física, equipo y geometría real del campo.

La V8 transforma ese pipeline en un sistema mucho más completo y verificable. Conserva la base de detección y tracking, pero añade:

- umbrales y validaciones diferentes por clase;
- confirmación de tracks y manejo más conservador de predicciones;
- recuperación visual de la pelota cuando YOLO deja de detectarla;
- reconstrucción offline de identidades físicas;
- clasificación de equipos basada principalmente en estructura visual y no sólo en color;
- detección y seguimiento de porterías;
- estabilización de cámara;
- segmentación independiente de la superficie de juego;
- registro geométrico de la cancha mediante homografías confiables;
- calibración automática, asistida o cargada desde archivo;
- separación explícita entre geometría local, provisional y global confiable;
- eventos de gol y eventos enriquecidos con equipo e identidad;
- exportación con una jerarquía de sistemas de coordenadas;
- suavizado e interpolación de trayectorias para la repetición;
- narración automática y generación de reportes HTML/PDF;
- archivos de diagnóstico y una suite de pruebas automatizadas.

En términos de tamaño, el código fuente pasó aproximadamente de **43 archivos Python y 4,016 líneas** en V1 a **69 archivos Python y 14,253 líneas** en V8, además de **5 archivos de pruebas con 1,424 líneas**. La suite incluida en V8 ejecutó correctamente **42 pruebas**.

---

## 3. Comparación general

| Área | V1 | V8 |
|---|---|---|
| Carga y validación de video | Presente | Se conserva prácticamente igual |
| Detección | Un modelo YOLO y umbral general | Modelo seleccionable, umbrales por clase, aliases y filtros geométricos |
| Robots | Hasta cuatro tracks temporales | Tracks confirmados, mejores reglas de asociación, identidad online y offline |
| Pelota | Track dedicado con refinamiento naranja | Track dedicado, modelo adaptativo de color y recuperación local |
| Cancha | Caja detectada y suavizada | Segmentación independiente, selección por cobertura y geometría registrada |
| Coordenadas | Píxeles o normalización respecto al video/caja | Homografía confiable, caja, cámara estabilizada o píxeles, en ese orden |
| Cámara móvil | No compensada explícitamente | Registro y estabilización de movimiento de cámara |
| Equipos | No disponibles | Clasificación online y reconstrucción offline |
| Porterías | No modeladas | Detección, IDs laterales y polígonos transformados |
| Eventos | Posesión, fuera, inactividad, pelota perdida y colisión | Eventos anteriores enriquecidos más goles y mayor control de confiabilidad |
| Replay | Movimiento directo sobre cancha esquemática | Suavizado, interpolación, visibilidad, equipos, porterías y panel mejorado |
| Depuración | Preview y JSONL de detecciones | Logs de rechazo, tracking, homografía, geometría y vistas rectificadas |
| Productos finales | JSON, gráficas y replay | Lo anterior más narración, subtítulos, video narrado y reporte PDF |
| Pruebas | No se encontró una suite equivalente | 42 pruebas automatizadas incluidas y aprobadas |

---

## 4. Cambios en la arquitectura del pipeline

### V1

La V1 organizaba el proceso principal en etapas para:

1. cargar y validar el video;
2. crear un preview con detecciones y tracking;
3. extraer eventos y trayectorias;
4. detectar opcionalmente la mano del árbitro mediante SAM;
5. exportar los datos para Unity/Mesa;
6. generar la repetición y el resumen.

La arquitectura ya estaba separada en módulos de pipeline, visión, dominio, eventos y simulación.

### V8

La V8 conserva esa organización básica, pero amplía el flujo con nuevas fases y configuraciones:

- selección entre pesos YOLO nuevos, legacy o personalizados;
- segmentación independiente de la cancha;
- registro geométrico y calibración;
- compensación de cámara;
- clasificación de equipos;
- reconstrucción offline de identidades;
- generación de diagnósticos geométricos;
- narración opcional;
- reporte HTML/PDF opcional.

El ejecutable principal también incorpora perfiles de rendimiento, opciones de depuración, parámetros de calibración, controles de interpolación y selección de motores de voz.

**Resultado:** el pipeline dejó de ser solamente un detector/tracker con replay y pasó a ser una cadena completa de análisis, validación geométrica, reconstrucción, presentación y reporte.

---

## 5. Detección de objetos

### Lo que ya existía en V1

- Uso de YOLO para detectar robots, pelota y cancha.
- Ejecución cuadro por cuadro.
- Conversión de detecciones al formato interno del tracker.
- Confianza global configurable.
- Modelo principal almacenado dentro del proyecto.

### Cambios en V8

#### 5.1 Umbrales diferentes por clase

En lugar de aplicar el mismo umbral a todos los objetos, V8 permite configurar valores independientes para:

- robots;
- pelota;
- cancha;
- porterías.

Esto permite ser más estricto con objetos grandes y frecuentes, como los robots, y más tolerante con objetos pequeños o difíciles, como la pelota.

#### 5.2 Normalización de nombres de clase

Se añadieron aliases en español e inglés para reconocer variantes de nombres como robot, ball, pelota, field, cancha, goal o portería. Esto reduce errores cuando el modelo cambia el nombre exacto de una clase.

#### 5.3 Filtros geométricos posteriores a YOLO

V8 no acepta automáticamente toda detección que supere la confianza. También revisa:

- ancho y alto mínimos;
- área mínima o máxima;
- proporción y tamaño razonables para cada clase;
- compatibilidad con la región de juego.

Las detecciones rechazadas se guardan junto con el motivo del rechazo, lo que facilita ajustar los parámetros.

#### 5.4 Nuevos modelos y selección de pesos

V8 incorpora:

- un nuevo modelo YOLO (`YOLOV2`);
- compatibilidad con el modelo legacy;
- posibilidad de indicar pesos personalizados;
- un modelo separado para segmentar la cancha.

#### 5.5 Optimización de ejecución

La versión nueva detecta si CUDA está disponible, puede utilizar precisión reducida y permite variar la resolución y frecuencia de inferencia según el perfil de rendimiento.

**Resultado:** la detección es más configurable, auditable y específica para las características de cada objeto.

---

## 6. Tracking temporal de robots

### Capacidades ya presentes en V1

Es importante aclarar que V1 no utilizaba únicamente una asociación por vecino más cercano. Ya incluía un tracker temporal relativamente avanzado con:

- máximo de cuatro robots;
- asignación global entre tracks y detecciones;
- costo basado en distancia, IoU, tamaño, apariencia y confianza;
- predicción de movimiento;
- memoria temporal para reidentificación;
- descriptor visual;
- un track separado para la pelota.

### Mejoras introducidas en V8

#### 6.1 Tracks tentativos y confirmados

Una detección nueva ya no se convierte inmediatamente en un robot definitivo. V8 exige varios aciertos consecutivos antes de confirmar el track. Esto reduce IDs creados por falsos positivos o reflejos momentáneos.

#### 6.2 Predicciones visibles más cortas

Los tracks pueden conservarse internamente durante una oclusión, pero las posiciones predichas sólo se muestran durante un número pequeño de cuadros. Así se evita dibujar robots en posiciones inventadas durante ausencias largas.

#### 6.3 Apariencia de referencia más estable

Además del descriptor reciente, V8 mantiene una referencia visual del robot para disminuir cambios de identidad cuando su apariencia temporal se altera por movimiento, iluminación u oclusiones.

#### 6.4 Asociación refinada

El costo de asociación se ajustó para considerar mejor:

- trayectoria predicha;
- desplazamientos imposibles;
- IoU y escala;
- apariencia reciente y de referencia;
- soporte dentro de la superficie de juego;
- calidad de la caja detectada.

La asignación global se conserva, pero con reglas más conservadoras.

#### 6.5 IDs restringidos y reutilización controlada

Los IDs online se mantienen dentro del conjunto esperado de cuatro robots. Si no existe un ID válido disponible, se rechaza la creación de otro track en lugar de aumentar indefinidamente la numeración.

#### 6.6 Diagnóstico del tracking

V8 genera un registro detallado con asociaciones, predicciones, estados, rechazos y razones de descarte. Esto permite saber por qué se mantuvo, perdió o cambió una identidad.

**Resultado:** el tracking online produce menos identidades falsas, limita la propagación de errores y proporciona evidencia para depurarlos.

---

## 7. Tracking y recuperación de la pelota

### V1

La pelota ya tenía un track independiente y un refinamiento local orientado al color naranja. Sin embargo, la recuperación estaba más ligada a la detección original y al comportamiento general del tracker.

### V8

Se creó una lógica especializada para la pelota:

- función de costo propia;
- aprendizaje adaptativo de tono, saturación, brillo y tamaño a partir de detecciones confiables;
- búsqueda local alrededor de la posición predicha;
- validación por color, área y proximidad;
- estado explícito `recuperado` cuando la pelota se encuentra por visión local;
- distinción entre medición real, recuperación visual y predicción pura.

Esto permite recuperar la pelota durante fallos cortos de YOLO sin confundir una estimación matemática con una observación visual.

**Resultado:** aumenta la continuidad del track de pelota, pero se mantiene trazabilidad sobre el origen de cada posición.

---

## 8. Identidad física y clasificación de equipos

Esta sección es completamente nueva en V8.

### 8.1 Clasificación online de equipos

V8 recolecta muestras visuales únicamente cuando el robot está claramente visible y la detección cumple condiciones de confianza y tamaño. Después intenta agrupar los cuatro robots en dos parejas.

Los descriptores priorizan:

- contornos y silueta;
- HOG y estructura de bordes;
- perfiles de ocupación;
- forma y topología;
- vistas rotadas para tolerar orientación.

El color tiene un peso bajo en la decisión de equipo para evitar que iluminación, reflejos o piezas similares dominen el agrupamiento.

Cuando el agrupamiento es ambiguo, el sistema conserva la etiqueta como desconocida en lugar de imponer una respuesta.

### 8.2 Reconstrucción offline de identidad

Después de procesar el video completo, V8 puede ejecutar una segunda pasada para corregir intercambios de ID:

1. divide los tracks online en segmentos o *tracklets*;
2. corta cuando hay saltos, huecos o cambios visuales incompatibles;
3. selecciona vistas representativas;
4. agrupa los segmentos en hasta cuatro identidades físicas;
5. impide asignar la misma identidad a segmentos que aparecen simultáneamente;
6. recupera tracklets cortos compatibles;
7. conserva como desconocidos los casos que no alcanzan suficiente evidencia;
8. reescribe las detecciones y trayectorias con ID físico, ID online original y equipo.

También se eliminan duplicados de una misma identidad dentro del mismo cuadro y se generan imágenes representativas para inspección.

### 8.3 Limitación semántica

La visión por sí sola puede formar dos equipos, pero no sabe cuál es “aliado” y cuál es “rival” sin una convención externa. V8 utiliza una regla determinista o un archivo de configuración para asignar esos nombres. Esta convención debe documentarse al interpretar resultados.

**Resultado:** V8 separa tres conceptos que en V1 estaban unidos: el track temporal, la identidad física del robot y su equipo.

---

## 9. Selección y segmentación de la cancha

### V1

La cancha se tomaba principalmente de las detecciones YOLO. Se elegía la caja con mayor confianza y se aplicaba suavizado y persistencia temporal. Esto ayudaba cuando la cámara era estable, pero podía mantener una región vieja cuando la cámara se movía o cuando la caja detectada era incompleta.

### V8

#### 9.1 Selección de la caja principal

La caja se elige combinando aproximadamente:

- cobertura de la imagen;
- confianza del detector.

Se eliminó la persistencia prolongada de la caja para no conservar una posición obsoleta cuando cambia la cámara.

#### 9.2 Segmentación independiente

Se añadió un modelo separado que produce:

- máscara de la superficie;
- polígono aproximado;
- cobertura de cancha;
- confianza y diagnósticos.

La segmentación puede ejecutarse cada cierto número de cuadros y reutilizarse entre inferencias para reducir costo.

#### 9.3 Soporte de superficie para objetos

Las detecciones de robots se evalúan respecto a la superficie estimada. Una caja sin apoyo suficiente dentro de la cancha puede rechazarse, evitando objetos del fondo o de los márgenes del video.

**Resultado:** la región de juego deja de depender únicamente de una caja YOLO y se convierte en evidencia geométrica independiente.

---

## 10. Estabilización de cámara

V8 incorpora un módulo de registro de cámara que no existía en V1.

El sistema:

- detecta características visuales sobre la superficie verde y las líneas blancas;
- excluye en lo posible robots y pelota;
- estima el movimiento entre cuadros;
- acumula una transformación respecto a un cuadro de referencia;
- rechaza desplazamientos o transformaciones físicamente inverosímiles;
- produce coordenadas de imagen estabilizadas y una medida de calidad.

Estas coordenadas no sustituyen a una homografía real de cancha, pero son un respaldo más estable que los píxeles originales cuando la cámara se mueve.

**Resultado:** el movimiento de cámara puede separarse parcialmente del movimiento real de los robots.

---

## 11. Geometría de cancha y homografía

Este es uno de los cambios centrales entre V1 y V8.

### V1

Las coordenadas utilizadas para exportación se calculaban principalmente de dos formas:

- normalización respecto al ancho y alto del video;
- posición relativa dentro de la caja rectangular detectada como cancha.

Esto era suficiente para una representación aproximada, pero no garantizaba que las posiciones coincidieran con la geometría física real.

### V8

#### 11.1 Plantilla canónica

Se define una cancha ideal, por defecto de 100 × 60 unidades, con:

- límites;
- línea central;
- áreas;
- arcos y marcas;
- zonas de portería.

#### 11.2 Fuentes de evidencia

La estimación combina:

- máscara segmentada;
- bordes físicos o rieles;
- líneas blancas;
- porterías;
- predicción temporal;
- calibraciones manuales o semánticas.

Los bordes de la imagen se descartan como evidencia de cancha para evitar confundir el encuadre del video con los límites físicos.

#### 11.3 Anclas semánticas

V8 reconoce líneas con significado, como:

- cercana;
- lejana;
- izquierda;
- derecha;
- central;
- líneas de área cercanas o lejanas.

Una combinación de dos líneas longitudinales y dos transversales puede determinar una homografía global, incluso cuando algunas son líneas internas y no se ven las cuatro esquinas externas.

#### 11.4 Geometría local contra global

La versión nueva diferencia explícitamente entre:

- **orientación local:** permite rectificar dirección o perspectiva, pero no ubicar globalmente el punto;
- **homografía provisional:** geométricamente plausible, pero sin evidencia suficiente;
- **homografía global confiable:** respaldada por anclas semánticas o estructura física válida.

Una calibración parcial no se presenta como coordenadas globales. Esto evita fabricar una ubicación absoluta a partir de información incompleta.

#### 11.5 Validación dura de candidatos

Las soluciones se puntúan por:

- error angular;
- perpendicularidad;
- correspondencia con la plantilla;
- longitud visible;
- error de reproyección;
- coherencia con porterías, máscara y líneas.

Una solución visualmente plausible puede permanecer sólo como diagnóstico si no cumple los requisitos de confianza.

#### 11.6 Propagación temporal

Una homografía global confiable puede propagarse durante intervalos cortos cuando falta segmentación, usando el registro de cámara. Una orientación local también puede propagarse, pero sigue marcada como local.

#### 11.7 Punto de contacto de los robots

Para transformar un robot se usa el centro inferior de su bounding box, que aproxima el punto donde toca la superficie. La pelota y otros objetos utilizan su centro.

**Resultado:** V8 deja de tratar cualquier rectificación plausible como una calibración válida y sólo publica coordenadas globales cuando existe evidencia suficiente.

---

## 12. Calibración asistida

V8 añade un asistente interactivo de calibración.

El usuario puede marcar únicamente las líneas semánticas visibles. El archivo generado conserva:

- resolución de referencia;
- etiquetas de cada línea;
- puntos seleccionados;
- tipo de calibración;
- solución global o local;
- métricas de calidad.

La calibración puede reutilizarse en otra resolución mediante escalamiento. También se mantiene compatibilidad con calibraciones completas de cuatro límites.

Modos disponibles:

- automático;
- asistido;
- archivo previamente generado.

**Resultado:** cuando la detección automática no dispone de suficiente evidencia, el usuario puede aportar restricciones interpretables en vez de seleccionar arbitrariamente cuatro esquinas.

---

## 13. Modelo de dominio

V8 amplía los objetos internos del sistema.

### Robot

Además de ID y bounding box, puede contener:

- equipo;
- número de equipo;
- nombre para mostrar;
- coordenadas estabilizadas;
- validez y calidad del registro de cámara;
- coordenadas físicas de cancha;
- coordenadas normalizadas;
- indicador de pertenencia a la superficie;
- fuente y confianza de la transformación;
- ID online original e ID físico corregido.

### Pelota

Añade información equivalente de estabilización y geometría, así como el estado medido, recuperado, interpolado o predicho.

### Portería

Se incorpora una clase nueva con:

- ID;
- lado visual izquierdo/derecho;
- estado visible o ausente;
- caja y polígono;
- coordenadas transformadas;
- prueba de si contiene el centro de la pelota.

### Estado del juego

El estado global ahora maneja aliases de clases, robots enriquecidos, pelota, porterías y metadatos geométricos.

---

## 14. Detección de eventos

### Eventos ya presentes en V1

- cambio de posesión;
- pelota fuera de la cancha;
- robot inactivo;
- pelota no detectada;
- posible colisión.

### Cambios en V8

#### 14.1 Eventos enriquecidos

Los eventos pueden utilizar:

- nombre físico del robot;
- equipo;
- confianza;
- fuente de coordenadas;
- método utilizado para decidir el evento.

#### 14.2 Pelota fuera

V8 primero consulta si la pelota está dentro de la superficie definida por la homografía. Si no existe una transformación confiable, utiliza la caja de cancha como respaldo. El evento registra qué método produjo la decisión.

#### 14.3 Gol

Se añade detección de gol con reglas temporales:

- pelota visible y medida, no sólo predicha;
- centro dentro de la portería;
- permanencia durante varios cuadros;
- periodo de rearme después de salir;
- confianza combinada de pelota y portería.

Cuando existe homografía confiable, puede identificarse la portería física cercana o lejana. La asignación del equipo anotador todavía depende de una convención temporal/configurable y no debe interpretarse como inferencia semántica absoluta.

**Resultado:** los eventos dependen más de observaciones confiables y documentan el método utilizado.

---

## 15. Sistemas de coordenadas y exportación

### V1

La exportación normalizaba las posiciones principalmente respecto al video. El replay convertía esas coordenadas a una cancha esquemática de 100 × 60.

### V8

Se establece la siguiente jerarquía:

1. coordenadas obtenidas con homografía global confiable;
2. coordenadas relativas a la caja de cancha;
3. coordenadas de cámara estabilizada;
4. píxeles originales del video.

Cada punto exportado incluye su `coordinate_source`, de modo que el consumidor sabe qué nivel de calidad está utilizando.

Las coordenadas moderadamente externas al campo no se recortan automáticamente a `[0,1]`. Esto permite representar correctamente una pelota fuera o dentro del volumen de una portería.

El esquema de exportación pasa de la versión 0.1 a la 0.2 e incluye:

- robots enriquecidos;
- porterías;
- nuevas fuentes de coordenadas;
- archivos de depuración;
- información de equipos e identidad.

**Resultado:** las coordenadas dejan de parecer equivalentes cuando en realidad provienen de métodos con distinta confiabilidad.

---

## 16. Repetición y simulación

### V1

La repetición Mesa mostraba robots y pelota moviéndose sobre una cancha esquemática a partir de los puntos exportados.

### V8

Se añadieron:

- nombres y equipos;
- colores y marcadores diferenciados;
- porterías y áreas canónicas;
- ocultamiento de agentes cuando no hay observación válida;
- suavizado bidireccional de trayectorias;
- limitación de saltos físicamente imposibles;
- interpolación curva de huecos de la pelota mediante Hermite;
- identificación explícita de puntos interpolados;
- panel de eventos mejorado;
- nombres de eventos en español;
- parámetros configurables para la duración máxima de interpolación.

El suavizado se aplica offline, de modo que puede utilizar información anterior y posterior sin alterar el tracker online.

**Resultado:** la repetición es más legible y continua, pero conserva la diferencia entre observaciones reales e interpolaciones.

---

## 17. Narración automática

V8 añade un módulo completo de narración que no estaba en V1.

Incluye:

- selección editorial de eventos;
- prioridades por tipo;
- eliminación de duplicados;
- periodos de enfriamiento;
- límites por categoría;
- lenguaje dependiente de la confianza;
- distribución temporal de los comentarios;
- generación de clips de voz;
- mezcla en una pista final;
- subtítulos SRT;
- video de muestra con audio y subtítulos.

Puede utilizar distintos motores, entre ellos Edge TTS, gTTS y opciones locales de Windows/Python.

---

## 18. Reporte HTML/PDF

También es nuevo en V8.

El reporte resume:

- duración analizada;
- número y tipo de eventos;
- eventos destacados;
- posesión;
- visibilidad de robots;
- identidad y equipo;
- visibilidad y estados de la pelota;
- trayectorias;
- mapa de calor de la pelota.

Se genera primero una plantilla HTML y después un PDF mediante WeasyPrint o Playwright como alternativa.

---

## 19. Rendimiento y configuración

V8 agrega perfiles de rendimiento:

| Perfil | Resolución aproximada de segmentación | Frecuencia aproximada |
|---|---:|---:|
| CPU | 448 px | cada 6 cuadros |
| Balanced | 512 px | cada 3 cuadros |
| Quality | 640 px | cada cuadro |
| Auto | Elige según hardware | Elige según hardware |

Los valores pueden sobrescribirse manualmente. Esto permite ejecutar una versión ligera en CPU o una versión de máxima calidad con GPU.

También se amplía la configuración para:

- modelos y pesos;
- thresholds por clase;
- calibración;
- equipos;
- identidad offline;
- estabilización;
- depuración;
- narración;
- reporte;
- interpolación.

---

## 20. Archivos de diagnóstico y salidas nuevas

### Salidas principales de V1

- `quick_preview.mp4`;
- `quick_detections.jsonl`;
- eventos, resumen y trayectorias en JSON;
- gráficas;
- exportación Unity/Mesa;
- video o GIF de replay;
- eventos opcionales de mano del árbitro.

### Salidas adicionales de V8

- detecciones rechazadas y motivos;
- debug completo del tracker;
- clustering de equipos online y offline;
- reconstrucción de identidad física;
- hoja de contactos de robots representativos;
- preview online y preview corregido offline;
- video de geometría de cancha;
- video rectificado;
- JSONL de homografías;
- archivo de calibración;
- manifest de narración;
- clips, pista final y subtítulos;
- video narrado de muestra;
- reporte HTML/PDF y gráficas asociadas.

---

## 21. Archivos y módulos añadidos en V8

Entre los módulos nuevos más importantes se encuentran:

| Módulo | Función |
|---|---|
| `ball_recovery.py` | Recuperación adaptativa de la pelota por color y proximidad |
| `field_selector.py` | Selección de la cancha principal por cobertura y confianza |
| `team_features.py` | Extracción de descriptores estructurales |
| `team_clustering.py` | Agrupamiento conservador en equipos |
| `team_classifier.py` | Clasificación online |
| `offline_identity.py` | Reconstrucción offline de identidades físicas |
| `goal.py` | Modelo de dominio para porterías |
| `field_registration.py` | Compensación del movimiento de cámara |
| `field_segmenter.py` | Segmentación independiente de la cancha |
| `field_template.py` | Plantilla canónica del campo |
| `field_geometry.py` | Gestión y validación de geometría |
| `feature_constraints.py` | Restricciones de líneas y anclas semánticas |
| `template_registration.py` | Registro automático contra la plantilla |
| `calibration.py` | Lectura, escalamiento y solución de calibración |
| `calibration_wizard.py` | Asistente interactivo |
| `G_narration/*` | Edición y síntesis de narración |
| `H_report/*` | Datos, gráficas y reporte PDF |
| `performance.py` | Perfiles de ejecución |

También se añadió `config/equipos.json`, el modelo `YOLOV2`, el modelo de segmentación y una carpeta de pruebas.

---

## 22. Componentes que se conservaron

No todo fue reemplazado. Los siguientes componentes permanecen esencialmente iguales o con cambios mínimos:

- carga y validación del video;
- análisis de metadatos;
- integración base con SAM para revisar la mano del árbitro;
- entidades geométricas básicas;
- estructura general por etapas;
- exportación conceptual hacia Mesa/Unity;
- generación de visualizaciones y resúmenes base.

Esto indica que la evolución se concentró en la interpretación visual y geométrica, no en reescribir innecesariamente las partes que ya funcionaban.

---

## 23. Validación automatizada

V8 incorpora pruebas para verificar, entre otros aspectos:

- estabilidad de IDs;
- confirmación y expiración de tracks;
- límite de predicciones visibles;
- recuperación adaptativa de pelota;
- reconstrucción de identidad después de un intercambio online;
- casos ambiguos de equipo;
- selección de cancha por cobertura;
- thresholds por clase;
- porterías y confirmación temporal de gol;
- segmentación y propagación geométrica;
- rechazo de bordes de imagen;
- calibración completa y parcial;
- escalamiento de calibración;
- solución con múltiples líneas;
- anclas semánticas internas;
- rechazo de correspondencias semánticas incorrectas;
- jerarquía de coordenadas del exportador;
- suavizado e interpolación del replay;
- perfiles de rendimiento.

Al ejecutar la suite incluida se obtuvieron **42 pruebas aprobadas**. Esto valida las unidades y escenarios cubiertos por el repositorio, aunque no sustituye una evaluación end-to-end con todos los videos y modelos reales.

---

## 24. Principales logros de la evolución

1. **Se mantuvo la base funcional de V1.** No se descartó el tracker existente; se fortaleció su confirmación, asociación y trazabilidad.
2. **Se separó detección de observación confiable.** Una caja YOLO, una recuperación visual y una predicción ya no se tratan como equivalentes.
3. **Se separó identidad online de identidad física.** Los intercambios temporales pueden corregirse después de observar el video completo.
4. **Se añadió interpretación de equipos sin depender principalmente del color.**
5. **Se dejó de asumir que el video es la cancha.** Ahora existen superficie, cámara estabilizada, geometría local y geometría global.
6. **Se evitó publicar homografías falsas.** Las coordenadas globales sólo se habilitan cuando existe evidencia estructural o semántica suficiente.
7. **Se incorporó recuperación ante oclusiones.** Tanto robots como pelota pueden mantener continuidad con controles explícitos.
8. **Se añadió soporte para goles y porterías.**
9. **Se mejoró la presentación final.** El replay, la narración y el reporte convierten los datos en productos interpretables.
10. **Se añadió observabilidad y pruebas.** Los errores pueden rastrearse mediante logs, videos de debug y escenarios automatizados.

---

## 25. Limitaciones que todavía deben considerarse

- La etiqueta aliado/rival requiere una convención o configuración; no puede inferirse con certeza sólo por apariencia.
- La atribución del equipo que anotó un gol sigue dependiendo de la convención de orientación y debe validarse para cada montaje.
- Una homografía provisional puede verse correcta sin ser globalmente válida; por eso V8 la mantiene como diagnóstico.
- La interpolación mejora el replay, pero no debe confundirse con medición real.
- La calidad de identidad offline depende de que existan suficientes vistas limpias y de que los robots tengan diferencias visuales observables.
- La narración y el PDF requieren dependencias opcionales externas.
- Las pruebas automatizadas cubren lógica interna, pero la validación final debe incluir videos representativos, oclusiones, cambios de cámara e iluminación.

---

## 26. Conclusión

La V1 estableció una base sólida de detección, tracking temporal, eventos y repetición. La V8 no es únicamente una versión con parámetros ajustados: representa una ampliación sustancial del sistema.

El cambio principal fue pasar de un pipeline que seguía objetos en la imagen a uno que intenta reconstruir el estado físico del partido de forma explícita y verificable. Para lograrlo se añadieron identidad offline, equipos, recuperación de pelota, estabilización, segmentación, calibración, homografía validada, porterías, nuevas reglas de eventos, jerarquía de coordenadas, mejores diagnósticos y productos finales de presentación.

En conjunto, V8 es más conservadora al decidir qué sabe, más detallada al registrar cómo lo sabe y mucho más completa al convertir detecciones visuales en coordenadas, eventos y reportes utilizables.
