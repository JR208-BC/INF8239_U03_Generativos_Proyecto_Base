"""Paso 7.5 de LAB10: comparar cada muestra generada por el VAE con su vecino
más cercano en el conjunto de entrenamiento (distancia euclidiana en píxeles),
para distinguir si el modelo copia ejemplos o produce variaciones genuinas.

Uso (desde la raíz del proyecto, después de correr train_ae_vae.py):
    uv run python scripts/nearest_neighbors.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

(x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = normalize_images(x_train)
flat_train = x_train.reshape(len(x_train), -1)

decoder = tf.keras.models.load_model(ROOT / "models/decoder.keras")
latent_points = np.random.default_rng(SEED).normal(size=(12, 2))
generated = decoder.predict(latent_points, verbose=0)
flat_generated = generated.reshape(len(generated), -1)

distances = np.linalg.norm(flat_train[None, :, :] - flat_generated[:, None, :], axis=2)
nearest_index = distances.argmin(axis=1)
nearest_distance = distances.min(axis=1)
nearest_images = x_train[nearest_index]

figure, axes = plt.subplots(2, 12, figsize=(12, 2.2))
for column in range(12):
    axes[0, column].imshow(generated[column].squeeze(), cmap="gray")
    axes[0, column].axis("off")
    axes[1, column].imshow(nearest_images[column].squeeze(), cmap="gray")
    axes[1, column].set_title(f"{nearest_distance[column]:.1f}", fontsize=7)
    axes[1, column].axis("off")
axes[0, 0].set_ylabel("Generada", fontsize=8)
figure.suptitle("Fila 1: generada · Fila 2: vecino más cercano en train (distancia L2 en título)", fontsize=8)
figure.tight_layout()
figure.savefig(ROOT / "reports/nearest_neighbors.png", dpi=170)

summary = {
    "distancia_l2_media": float(nearest_distance.mean()),
    "distancia_l2_min": float(nearest_distance.min()),
    "distancia_l2_max": float(nearest_distance.max()),
    "nota": "Distancia 0 indicaría copia exacta de un ejemplo de entrenamiento; valores altos sugieren variación genuina.",
}
(ROOT / "reports/nearest_neighbors.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Guardado: reports/nearest_neighbors.png y reports/nearest_neighbors.json")
