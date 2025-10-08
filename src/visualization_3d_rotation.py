### Run this visualization to get a rotation plot in 3D (RGB)
# Here we use: images from 'data' as original, 'out_data' for FancyGNG and from 'out_data_pca' for FancyPCA ###

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
from io import BytesIO

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
#image2 = Image.open("/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung/out_fancy_pca/heißluftballon2_1.jpg")

# Verkleinern der Bilder (falls nötig)
image1 = resize_image(image1)
#image2 = resize_image(image2)

# Konvertiere die Bilder in NumPy-Arrays
image1_array = np.array(image1)
#image2_array = np.array(image2)

# Wandle die Bilder in 2D-Arrays um (Anzahl der Pixel, 3 Farbkanäle)
image1_pixels = image1_array.reshape(-1, 3)
#image2_pixels = image2_array.reshape(-1, 3)

# Sampling der Pixel (SAMPLE_SIZE_FACTOR-% der Pixel für effizientere t-SNE-Berechnung)
sample_size1 = int(SAMPLE_SIZE_FACTOR * len(image1_pixels))  # Berechne die Anzahl der zu samplenden Pixel für Bild 1

# Zufällige Auswahl der Pixel - image1_pixels.shape[0]: Anzahl der Zeilen, sample_size1: Anzahl der zu ziehenden Stichproben, replace=False: keine doppelten Stichproben
image1_pixels_sampled = image1_pixels[np.random.choice(image1_pixels.shape[0], sample_size1, replace=False)] 

#sample_size2 = int(SAMPLE_SIZE_FACTOR * len(image2_pixels))  # Berechne die Anzahl der zu samplenden Pixel für Bild 2
#image2_pixels_sampled = image2_pixels[np.random.choice(image2_pixels.shape[0], sample_size2, replace=False)] # zufällige Auswahl der Pixel

### --- 3D-Plot erstellen --- ###
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Punkte plotten
ax.scatter(image1_pixels_sampled[:, 0], image1_pixels_sampled[:, 1], image1_pixels_sampled[:, 2], c=image1_pixels_sampled / 255.0, s=1)

# Achsenbeschriftungen
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

### --- Animation erstellen --- ###
num_frames = 40
angles = np.linspace(0, 360, num_frames)
frames = []

for angle in angles:
    ax.view_init(30, angle)  # Kamera-Rotation
    plt.draw()

    # Bild direkt aus dem Speicher speichern
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(imageio.imread(buf))

# GIF speichern
imageio.mimsave('rotation_rgb_plot.gif', frames, duration=0.05)

# Plot anzeigen
plt.show()