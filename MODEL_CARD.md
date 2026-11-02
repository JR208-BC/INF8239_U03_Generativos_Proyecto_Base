# Model Card · Autoencoder y VAE

## Modelos y versiones

- **Autoencoder (AE):** `build_autoencoder` en `src/inf8239_u03_gen/models.py` — codificador/decodificador densos, espacio latente de 16 dimensiones. Entrenado 6 épocas (con `EarlyStopping`, paciencia 2). Archivo: `models/autoencoder.keras` (332 KB, 25,888 parámetros).
- **VAE:** `build_encoder_decoder` + clase `VAE` (entrenamiento personalizado con `train_step`) en `scripts/train_ae_vae.py` — espacio latente de **2 dimensiones** (elegido deliberadamente bajo, para poder visualizar e interpolar directamente). Entrenado 8 épocas. Archivo evaluado: `models/decoder.keras` (416 KB, 101,520 parámetros).
- Versión: noviembre de 2026, semilla 42, `TF_DETERMINISTIC_OPS=1`.
- Entorno: Python 3.13, TensorFlow 2.21.
- Responsable: Jose Berroa, curso INF-8239, Unidad 03.

## Uso previsto y usuarios

- Uso académico: comparar un autoencoder determinista con un autoencoder variacional (VAE) sobre el mismo dataset y la misma arquitectura base, para entender reconstrucción vs. generación.
- Explorar interactivamente el espacio latente de 2 dimensiones del VAE (`app/streamlit_app.py`).
- Estudiantes e instructores del curso; no hay usuarios finales reales.

## Usos fuera de alcance

- Generar imágenes de prendas reales para catálogo, publicidad o cualquier uso comercial.
- Cualquier uso con fotos de personas o datos sensibles — el dataset son solo prendas de catálogo en escala de grises.
- Presentar una imagen generada como foto real de un producto existente.
- Afirmar privacidad o anonimato por el solo hecho de que una imagen sea "sintética": una muestra generada puede seguir siendo muy similar a un ejemplo de entrenamiento (ver sección de vecinos cercanos).
- Producción o cualquier decisión automatizada.

## Dataset, licencia y particiones

- Dataset: Fashion-MNIST (Han Xiao, Kashif Rasul y Roland Vollgraf, Zalando Research, 2017; arXiv:1708.07747). Licencia MIT.
- 60,000 imágenes de entrenamiento, 10,000 de prueba, 28×28 en escala de grises, 10 clases de prendas.
- Preprocesamiento: conversión a `float32`, escalado de 0–255 a 0–1, canal agregado (28×28×1); validado con `validate_images()`.
- Particiones: AE usa `validation_split=0.1` sobre el entrenamiento; VAE entrena sobre el conjunto completo de entrenamiento (sin validación explícita, al no tener una métrica de validación estándar para pérdida ELBO en este script). Prueba (10,000 imágenes) reservada para la evaluación final (paneles, interpolación, vecinos cercanos).

## Arquitectura y espacio latente

- **AE:** `Flatten → Dense(16, relu) [latente] → Dense(784, sigmoid) → Reshape(28,28,1)`. Un solo vector latente determinista por imagen.
- **VAE:** codificador `Flatten → Dense(128, relu) → [Dense(2) media, Dense(2) log-varianza]`; muestreo con el truco de reparametrización (`z = media + exp(0.5·log_var)·ε`, `ε ~ N(0,1)`); decodificador `Dense(128, relu) → Dense(784, sigmoid) → Reshape(28,28,1)`.
- Pérdida del VAE: `reconstrucción (entropía cruzada binaria, suma por imagen) + KL (regularización hacia N(0,1))`. Al final del entrenamiento: pérdida total 273.59, reconstrucción 267.33, KL 6.26 (última época).
- El espacio latente del VAE es de solo 2 dimensiones — suficiente para visualizar directamente (ejes X/Y en la app), pero limita la cantidad de variación que el modelo puede representar.

## Métricas de reconstrucción y generación

- AE: pérdida de validación final (entropía cruzada binaria) 0.3071, bajando de forma estable en las 6 épocas — sin señal de sobreajuste en ese tramo corto.
- VAE: pérdida de reconstrucción 267.33 (de un máximo teórico mucho mayor dado 784 píxeles), KL 6.26 — el componente KL es pequeño comparado con la reconstrucción, indicando que el modelo prioriza reconstruir bien sobre regularizar fuertemente el espacio latente.
- Panel comparativo (`reports/ae_vae_panel.png`, 12 ejemplos fijos del conjunto de prueba, sin selección manual): fila 1 originales, fila 2 reconstrucciones del AE, fila 3 reconstrucciones del VAE, fila 4 muestras generadas desde puntos aleatorios `N(0,1)` del espacio latente.
  - Las reconstrucciones del AE son más nítidas que las del VAE (el AE no tiene presión de regularización KL).
  - Las reconstrucciones del VAE conservan la silueta general (zapato, pantalón, camisa) pero con más borrosidad.
  - Las muestras **generadas** (fila 4) son notablemente más borrosas y, en varios casos, mezclan rasgos de más de una clase (p. ej. columnas que no se identifican claramente como una prenda específica) — limitación esperable de un espacio latente de solo 2 dimensiones con una arquitectura densa simple.
- Interpolación (`reports/interpolation.png`, 12 pasos lineales entre dos puntos del espacio latente): la transición es gradual y plausible — de una forma tipo bota/zapato se pasa, a través de formas intermedias borrosas, a una forma tipo pantalón, sin saltos abruptos.

## Diversidad y vecinos cercanos

- Comparación con `scripts/nearest_neighbors.py` (no incluido en el proyecto base; se agregó para cubrir el punto 5 del paso 7 del laboratorio): para 12 muestras generadas, se buscó el vecino más cercano en el conjunto de entrenamiento (distancia euclidiana en píxeles).
- Distancia L2 media 2.33 (rango 1.64–3.54) — **ninguna distancia es cero**, es decir, el modelo no está copiando literalmente ejemplos de entrenamiento. Al mismo tiempo, las distancias no son muy grandes: las muestras generadas son variaciones borrosas de prendas existentes, no diseños completamente novedosos.
- No se observó colapso total (todas las muestras generadas distintas entre sí), pero sí baja diversidad de detalle fino, consistente con la limitación de un espacio latente de 2 dimensiones.

## Hardware, tiempo y tamaño

- Hardware: contenedor Linux, solo CPU, TensorFlow 2.21.0, Python 3.13.
- Tiempo de entrenamiento: AE 9.5 s (6 épocas), VAE 19.0 s (8 épocas) — el VAE cuesta ~2× más tiempo que el AE por el paso de muestreo y el cálculo adicional de la pérdida KL.
- Tamaño de los artefactos: `autoencoder.keras` 332 KB (25,888 parámetros), `decoder.keras` 416 KB (101,520 parámetros) — el decodificador del VAE es más grande porque su primera capa densa parte de solo 2 dimensiones latentes hacia 128 unidades, mientras que el AE usa 16 dimensiones latentes directamente hacia la salida.

## Limitaciones, riesgos y supervisión

**Limitaciones**
- Arquitecturas completamente densas (sin convoluciones): no aprovechan la estructura espacial 2D de las imágenes, lo que limita la nitidez alcanzable frente a una CNN equivalente.
- Espacio latente del VAE de solo 2 dimensiones: suficiente para visualizar, insuficiente para capturar toda la variación del dataset (10 clases de prendas).
- Entrenamiento corto (6 y 8 épocas) con fines de verificación rápida; con más épocas las pérdidas probablemente seguirían bajando.
- Fashion-MNIST es un benchmark didáctico: 28×28, escala de grises, prendas de catálogo centradas — no representa fotos reales de ropa en contextos variados.

**Riesgos**
- Interpretar una imagen generada borrosa o ambigua como si representara una prenda real existente.
- Asumir que "sintético" implica "privado" u "original" — ya se demostró que las muestras generadas son variaciones cercanas (no idénticas, pero sí próximas) a ejemplos reales de entrenamiento.
- Usar el espacio latente de 2 dimensiones como si tuviera un significado semántico garantizado en sus ejes (por ejemplo, asumir que el eje X siempre representa "tipo de calzado") sin verificarlo explícitamente.

**Supervisión**
- Revisión humana de cualquier imagen generada antes de usarla fuera de este ejercicio académico.
- No desplegar este modelo como generador de contenido real sin reentrenar con una arquitectura convolucional, más épocas, y una evaluación de diversidad más rigurosa (por ejemplo, FID u otra métrica de distribución, no solo inspección visual).
- Mantener la advertencia visible en la aplicación de que una muestra sintética no garantiza privacidad ni originalidad.
