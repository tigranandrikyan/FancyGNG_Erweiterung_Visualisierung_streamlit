import numpy as np
import matplotlib.pyplot as plt

# Metriken für die Berechnung von Bildähnlichkeiten
from skimage.metrics import structural_similarity as ssim # berechnet strukturelle Ähnlichkeit -> 1: vollständig identisch; -1: maximal unterschiedlich
from sklearn.metrics import mean_squared_error # 0: Bilder sind identisch; Je höher der Wert, desto unterschiedlicher die Bilder

from PIL import Image

# Funktion zum Laden eines Bildes
def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    return image

# Funktion zum Berechnen der Metriken
def calculate_metrics(image1, image2):
    # Berechnung des MSE (Mean Squared Error)
    mse_value = mean_squared_error(image1.flatten(), image2.flatten())
    
    # Berechnung des SSIM (Structural Similarity Index)
    try:
        ssim_value = ssim(image1, image2, multichannel=True, win_size=3, channel_axis=2)
    except ValueError:
        print("Fehler bei SSIM: Möglicherweise sind die Bilder zu klein oder das win_size ist zu groß.")
        ssim_value = None

    return mse_value, ssim_value

image1_path = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/data/adler.jpg"  # Pfad zum ersten Bild (Original)
image2_path = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/out_data/adler_1.jpg"  # Pfad zum zweiten Bild (FancyGNG)
image3_path = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/out_fancy_pca/adler_1.jpg"  # Pfad zum dritten Bild (FancyPCA)

# Lade die Bilder
image1 = load_image(image1_path)
image2 = load_image(image2_path)
image3 = load_image(image3_path)

# Sicherstellen, dass alle Bilder die gleiche Größe haben
if image1.shape != image2.shape:
    #print("Die Bilder haben unterschiedliche Größen! Sie werden nun auf die gleiche Größe skaliert.")
    image2 = Image.fromarray(image2).resize(image1.shape[1::-1])
    image2 = np.array(image2)

if image1.shape != image3.shape:
    #print("Die Bilder haben unterschiedliche Größen! Sie werden nun auf die gleiche Größe skaliert.")
    image3 = Image.fromarray(image3).resize(image1.shape[1::-1])
    image3 = np.array(image3)

# Berechne die Metriken für jedes Bildpaar
mse_value_1_2, ssim_value_1_2 = calculate_metrics(image1, image2)
mse_value_1_3, ssim_value_1_3 = calculate_metrics(image1, image3)

# Ausgabe der Ergebnisse
print(f"\nVergleich zwischen Bild 1 und Bild 2:")
print(f"Mean Squared Error (MSE): {mse_value_1_2}")
if ssim_value_1_2 is not None:
    print(f"Structural Similarity Index (SSIM): {ssim_value_1_2}")

print(f"\nVergleich zwischen Bild 1 und Bild 3:")
print(f"Mean Squared Error (MSE): {mse_value_1_3}")
if ssim_value_1_3 is not None:
    print(f"Structural Similarity Index (SSIM): {ssim_value_1_3}")

# Zeige die Bilder 2 und 3 an mit den Metriken unterhalb
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Bild 1 anzeigen
axes[0].imshow(image1)
axes[0].set_title("Bild 1 (Original)")
axes[0].axis("off")

# Bild 2 anzeigen und Metriken darunter anzeigen
axes[1].imshow(image2)
axes[1].set_title("Bild 2 (FancyGNG)")
axes[1].axis("off")
axes[1].text(0.5, -0.1, f"MSE zum Original: {mse_value_1_2:.4f}\nSSIM zum Original: {ssim_value_1_2:.4f}", ha='center', va='center', transform=axes[1].transAxes)

# Bild 3 anzeigen und Metriken darunter anzeigen
axes[2].imshow(image3)
axes[2].set_title("Bild 3 (FancyPCA)")
axes[2].axis("off")
axes[2].text(0.5, -0.1, f"MSE zum Original: {mse_value_1_3:.4f}\nSSIM zum Original: {ssim_value_1_3:.4f}", ha='center', va='center', transform=axes[2].transAxes)

# Layout anpassen und Bilder anzeigen
plt.tight_layout()
plt.show()