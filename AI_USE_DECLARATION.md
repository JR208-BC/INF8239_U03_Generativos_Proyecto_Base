# Declaración de uso de inteligencia artificial

## Herramientas utilizadas

Claude (Anthropic), como asistente de código, depuración y redacción.

## Propósito de cada uso

- Revisión general del proyecto antes de armar el Ejercicio 06 (código, pruebas, `.gitignore`, documentación).
- Ejecución del entrenamiento (`scripts/train_ae_vae.py`) y generación de la evidencia (panel comparativo, interpolación) cuando el proyecto todavía no tenía reportes.
- Construcción de un script adicional, no incluido en el proyecto base, para cubrir el punto 5 del paso 7 de LAB10 ("Compare con vecinos cercanos del entrenamiento"): `scripts/nearest_neighbors.py`.
- Redacción de `MODEL_CARD.md` y de esta declaración, a partir de los resultados numéricos y las imágenes generadas.

## Prompts o decisiones relevantes

- Se pidió revisar el `.gitignore` (igual que en los ejercicios anteriores de esta unidad, excluía `reports/*.png`, `reports/*.json` y `models/*.keras`, impidiendo que la evidencia quedara en el repositorio) y corregirlo.
- Se pidió generar `requirements.txt` con `uv export`, exigido por el mandato y ausente del proyecto base.
- Se pidió interpretar el panel AE/VAE/generación y la interpolación con criterio crítico: distinguir nitidez de reconstrucción vs. plausibilidad de generación, y señalar explícitamente la borrosidad y la posible mezcla de clases en las muestras generadas, sin exagerar la calidad del resultado.
- Se pidió diseñar la comparación de vecinos más cercanos (distancia euclidiana en píxeles entre cada muestra generada y el ejemplo de entrenamiento más parecido) para poder afirmar, con evidencia numérica y no solo visual, que el modelo no copia literalmente ejemplos de entrenamiento.

## Elementos verificados por el equipo

- Se ejecutó personalmente `uv run pytest -q` (4 pruebas) y `scripts/train_ae_vae.py` en el propio equipo, y se compararon los valores de pérdida y tiempo obtenidos contra los documentados aquí.
- Se revisaron visualmente `reports/ae_vae_panel.png`, `reports/interpolation.png` y `reports/nearest_neighbors.png` antes de aceptar la interpretación escrita por la IA.
- Se verificó que ninguna distancia L2 en `reports/nearest_neighbors.json` fuera cero (lo que habría indicado copia exacta) antes de afirmar que el modelo genera variaciones, no copias.

## Cambios realizados sobre resultados generados

- Ninguna predicción, imagen o métrica fue editada manualmente para mejorar su apariencia; los valores reportados en `MODEL_CARD.md` y en el PDF de entrega provienen directamente de `reports/training_metrics.json` y `reports/nearest_neighbors.json`.

## Responsabilidad asumida

Asumo la responsabilidad por los datos, el código, las referencias y las conclusiones de esta entrega, y puedo explicar cualquier fragmento presentado, incluyendo el script adicional de vecinos más cercanos y las decisiones de arquitectura del AE y el VAE.
