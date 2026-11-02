# INF-8239 · Unidad 03 · Ejercicio 06 — Autoencoder, VAE y Model Card

Autor: Jose Rafael Berroa Castillo

¿Qué diferencia hay entre reconstruir imágenes (autoencoder) y generar imágenes nuevas y plausibles (VAE), y qué tan reales/originales son realmente las muestras generadas?

- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research, licencia MIT). 60,000 imágenes de entrenamiento, 10,000 de prueba, 28×28 en escala de grises.
- **Resultado:** el AE (16 dimensiones latentes) reconstruye con nitidez (pérdida de validación 0.307); el VAE (2 dimensiones latentes) reconstruye con más borrosidad pero permite generar e interpolar. Las muestras generadas al azar son notablemente más borrosas que las reconstrucciones y a veces mezclan rasgos de más de una clase.
- **Vecinos más cercanos:** ninguna muestra generada coincide exactamente con un ejemplo de entrenamiento (distancia L2 media 2.33, mínima 1.64) — el modelo no copia, pero tampoco produce diseños muy alejados de lo visto.
- **Aviso:** ejercicio académico; una imagen sintética no es automáticamente privada, original ni libre de sesgo — ver `MODEL_CARD.md`.

## Instalación

```bash
uv python install 3.12
uv sync --extra cpu     # CPU, recomendado para todos los estudiantes
# uv sync --extra gpu    # solo WSL2/Linux con GPU NVIDIA ya configurada
uv run python scripts/check_runtime.py
```

## Entrenamiento y evaluación

```bash
uv run pytest -q                                           # 4 pruebas
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
uv run python scripts/nearest_neighbors.py                 # comparación con vecinos más cercanos
```

Genera `reports/ae_vae_panel.png` (originales / reconstrucción AE / reconstrucción VAE / generación), `reports/interpolation.png`, `reports/training_metrics.json`, `reports/nearest_neighbors.png` y `reports/nearest_neighbors.json`, además de `models/autoencoder.keras` y `models/decoder.keras`.

## Resultados principales

| Modelo | Espacio latente | Parámetros | Tiempo de entrenamiento | Tamaño |
|---|---|---|---|---|
| Autoencoder | 16 | 25,888 | 9.5 s (6 épocas) | 332 KB |
| VAE (decodificador) | 2 | 101,520 | 19.0 s (8 épocas) | 416 KB |

Interpretación completa (reconstrucción, generación, interpolación, vecinos más cercanos, costo y limitaciones) en `MODEL_CARD.md` y `CIERRE_INTERPRETATIVO.md`.

## Demostración local

```bash
uv run streamlit run app/streamlit_app.py
```

Explora el espacio latente 2D del VAE con dos sliders; avisa explícitamente que una muestra sintética no garantiza originalidad ni privacidad.

## Estructura

```
src/inf8239_u03_gen/   código reutilizable (normalización/validación de imágenes, arquitecturas AE y VAE)
scripts/               verificación de entorno, entrenamiento, vecinos más cercanos
app/                   demostración local en Streamlit (explorador del espacio latente)
tests/                 pruebas con pytest (contrato de datos y de los modelos)
reports/               panel comparativo, interpolación, métricas, vecinos más cercanos
models/                autoencoder y decodificador del VAE serializados
```

## Decisiones clave

- **Semilla fija (42) y `TF_DETERMINISTIC_OPS=1`** en entrenamiento y muestreo, para reproducibilidad.
- **Espacio latente del VAE en 2 dimensiones:** deliberadamente bajo para poder visualizar e interpolar directamente, a costa de capacidad representacional (ver limitaciones en `MODEL_CARD.md`).
- **Panel sin selección manual:** los 12 ejemplos del panel comparativo son los primeros del conjunto de prueba (orden fijo, no elegidos por calidad del resultado).
- **Vecinos más cercanos:** se agregó esta comprobación (no estaba en el proyecto base) porque el laboratorio la exige explícitamente y es la única forma de verificar con evidencia, no solo visualmente, que el modelo no memoriza ejemplos de entrenamiento.

## Herramientas de IA utilizadas

Ver `AI_USE_DECLARATION.md`.

## Licencia y cita

Dataset bajo licencia MIT. Cita: Xiao, H., Rasul, K. y Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
