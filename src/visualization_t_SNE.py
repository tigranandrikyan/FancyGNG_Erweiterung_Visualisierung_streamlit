### Run this visualization to get two t_SNE plots for chosen images 
# Here we use: images from 'data' as original, 'out_data' for FancyGNG and from 'out_data_pca' for FancyPCA ###

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Definiere globale Parameter 'MAX_PIXELS' für 'resize_image' und 'SAMPLE_SIZE_FACTOR' für 'sample_size1' und 'sample_size2' (Bildverarbeitung)
MAX_PIXELS = 10000  # Maximale Anzahl von Pixeln für die Bildverkleinerung, z.B. 10000 für eine schnellere Berechnung
SAMPLE_SIZE_FACTOR = 0.1  # Faktor für die Auswahl von 10% der Pixel (z.B. 0.1 für 10%) für eine schnelle t-SNE-Berechnung

# Funktion zum Verkleinern des Bildes, falls es zu groß ist
def resize_image(image, max_pixels=MAX_PIXELS):
    width, height = image.size # Extrahiere die aktuelle Breite und Höhe des Bildes
    print(f"Ursprüngliche Bildgröße: {width}x{height}")  # Originalgröße ausgeben für Debugging-Zwecke
    total_pixels = width * height # Berechne die Gesamtanzahl der Pixel im Bild
    
    if total_pixels > max_pixels:  # Falls das Bild zu groß ist
        scale_factor = (max_pixels / total_pixels) ** 0.5  # Wurzel ziehen für Skalierung
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Verkleinere das Bild unter Verwendung des LANCZOS-Filters für eine hohe Qualität
        image = image.resize((new_width, new_height), Image.LANCZOS)  # Verwendung von Image.LANCZOS
        print(f"Bild verkleinert auf: {new_width}x{new_height}") # Debugging-Ausgabe für die neue Größe
    else:
        print(f"Bildgröße ist OK: {width}x{height}") # Debugging-Ausgabe für die Größe, falls das Bild nicht verkleinert wurde

    return image # Rückgabe des (ggf. verkleinerten) Bildes

# Bild laden und ggf. verkleinern
image1 = Image.open("/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/out_data/blumenwiese2_1.jpg")
image2 = Image.open("/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/out_fancy_pca/heißluftballon2_1.jpg")

# Verkleinern der Bilder (falls nötig)
image1 = resize_image(image1)
image2 = resize_image(image2)

# Konvertiere die Bilder in NumPy-Arrays
image1_array = np.array(image1)
image2_array = np.array(image2)

# Wandle die Bilder in 2D-Arrays um (Anzahl der Pixel, 3 Farbkanäle)
image1_pixels = image1_array.reshape(-1, 3)
image2_pixels = image2_array.reshape(-1, 3)

# Sampling der Pixel (SAMPLE_SIZE_FACTOR-% der Pixel für effizientere t-SNE-Berechnung)
sample_size1 = int(SAMPLE_SIZE_FACTOR * len(image1_pixels))  # Berechne die Anzahl der zu samplenden Pixel für Bild 1

# Zufällige Auswahl der Pixel - image1_pixels.shape[0]: Anzahl der Zeilen, sample_size1: Anzahl der zu ziehenden Stichproben, replace=False: keine doppelten Stichproben
image1_pixels_sampled = image1_pixels[np.random.choice(image1_pixels.shape[0], sample_size1, replace=False)] 

sample_size2 = int(SAMPLE_SIZE_FACTOR * len(image2_pixels))  # Berechne die Anzahl der zu samplenden Pixel für Bild 2
image2_pixels_sampled = image2_pixels[np.random.choice(image2_pixels.shape[0], sample_size2, replace=False)] # zufällige Auswahl der Pixel

# Initialisiere t-SNE mit 2 Dimensionen für die Visualisierung
tsne = TSNE(n_components=2, perplexity=30, random_state=42) ### perplexity-Wert anpassen, falls Bild sehr stark verkleinert wird (z.B. bei 12x8 * SAMPLE_SIZE_FACTOR Pixel < perplexity) ###

# Berechne die t-SNE-Transformation für die gesampelten Pixel
image1_tsne = tsne.fit_transform(image1_pixels_sampled) # Transformation für Bild 1
image2_tsne = tsne.fit_transform(image2_pixels_sampled) # Transformation für Bild 2

# Erstelle die Subplots (zwei t-SNE-Plots nebeneinander)
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Erster t-SNE-Plot für Bild 1 (Farben der gesampelten Pixel)
axs[0, 0].scatter(image1_tsne[:, 0], image1_tsne[:, 1], c=image1_pixels_sampled / 255, s=1) # Punkte mit Originalfarben plotten
axs[0, 0].set_title("t-SNE Farbvisualisierung Bild 1 (Sampling)")
axs[0, 0].axis("off")

# Zweiter t-SNE-Plot für Bild 2
axs[0, 1].scatter(image2_tsne[:, 0], image2_tsne[:, 1], c=image2_pixels_sampled / 255, s=1)
axs[0, 1].set_title("t-SNE Farbvisualisierung Bild 2 (Sampling)")
axs[0, 1].axis("off")

# Verpixeltes Bild 1 unter dem ersten t-SNE-Plot
axs[1, 0].imshow(image1)
axs[1, 0].axis("off") # Achsen ausblenden
axs[1, 0].set_title("Skaliertes Bild 1")

# Verpixeltes Bild 2 unter dem zweiten t-SNE-Plot
axs[1, 1].imshow(image2)
axs[1, 1].axis("off") # Achsen ausblenden
axs[1, 1].set_title("Skaliertes Bild 2")

# Zeige die gesamten Subplots
plt.tight_layout()
plt.show()