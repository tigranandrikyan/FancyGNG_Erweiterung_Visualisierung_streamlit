### Run this visualization to get three figures: original, FancyGNG and FancyPCA and the colour plots in the RG-, RB- and GB-space each and save them for comparison reason
# note: close the figure to come to the next figure ###

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Liste der Bildpfade
image_paths = [
    "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/data/adler.jpg", # Originalbild
    "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/out_data/adler_1.jpg", # FancyGNG
    "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/out_fancy_pca/adler_1.jpg" # FancyPCA
]

# Speicherpfad für die Plots
save_dir = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung"
os.makedirs(save_dir, exist_ok=True) # falls save_dir nicht existiert, wird er erstellt; exist_ok=True: wenn der Ordner bereits existiert, dann passiert nichts

# **Verarbeitung für jedes Bild**
for i, image_path in enumerate(image_paths, start=1):
    # Öffne das Bild
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Extrahiere die RGB-Werte
    r_values, g_values, b_values, colors = [], [], [], []

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            r_values.append(r)
            g_values.append(g)
            b_values.append(b)
            colors.append((r / 255, g / 255, b / 255))  # Normierte Farben für Scatterplots

    # Konvertiere in Numpy-Arrays
    r_values = np.array(r_values)
    g_values = np.array(g_values)
    b_values = np.array(b_values)

    # **Neue Figur für jedes Bild**
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # RG-Plot (links oben)
    axes[0, 0].scatter(r_values, g_values, c=colors, s=1)
    axes[0, 0].set_xlabel("R")
    axes[0, 0].set_ylabel("G")
    axes[0, 0].set_xlim(0, 255)
    axes[0, 0].set_ylim(0, 255)
    axes[0, 0].set_title("RG-Farbverteilung")

    # RB-Plot (rechts oben)
    axes[0, 1].scatter(r_values, b_values, c=colors, s=1)
    axes[0, 1].set_xlabel("R")
    axes[0, 1].set_ylabel("B")
    axes[0, 1].set_xlim(0, 255)
    axes[0, 1].set_ylim(0, 255)
    axes[0, 1].set_title("RB-Farbverteilung")

    # GB-Plot (rechts unten)
    axes[1, 1].scatter(g_values, b_values, c=colors, s=1)
    axes[1, 1].set_xlabel("G")
    axes[1, 1].set_ylabel("B")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_ylim(0, 255)
    axes[1, 1].set_title("GB-Farbverteilung")

    # RGB-Bild (links unten)
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Bild (RGB)")
    axes[1, 0].axis("off")

    # Layout-Anpassung
    plt.tight_layout()

    # **Zeige die Figur an und warte auf manuelles Schließen**
    plt.show(block=True)

    # **Speichern der Figur**
    save_path = os.path.join(save_dir, f"adler_{i}_rgb.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figur gespeichert unter: {save_path}")