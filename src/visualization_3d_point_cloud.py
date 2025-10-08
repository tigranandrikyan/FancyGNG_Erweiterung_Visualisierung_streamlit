### Run this visualization to get a 2D point cloud plot ###

# Importiere benötigte Bibliotheken

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Lade die beiden Bilder
image1_path = "/Users/macbookair/Desktop/Materialien/Python/DBL-GNG-main/Vektorquantisierung_3dplot_amir/Mohn_Field_1.jpeg"
image2_path = "/Users/macbookair/Desktop/Materialien/Python/DBL-GNG-main/Vektorquantisierung_3dplot_amir/Rapsfeld_2007.jpeg"
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Skaliere die Bilder auf die gleiche Größe
width, height = min(image1.size[0], image2.size[0]), min(image1.size[1], image2.size[1]) # Skalierung auf eine gemeinsame Größe, bei der das kleinere Bild die Größe vorgibt
image1 = image1.resize((width, height)) # Das verkleinerte Bild wird 'image1' zugewiesen
image2 = image2.resize((width, height)) # Das verkleinerte Bild wird 'image2' zugewiesen

# Konvertiere die Bilder in RGB-Punktewolken
rgb_image1 = image1.convert("RGB")
rgb_image2 = image2.convert("RGB")

# Erstelle eine Liste von 3D-Punkten und Farben für beide Bilder
points1 = []
points2 = []
for x in range(width):
    for y in range(height):
        r1, g1, b1 = rgb_image1.getpixel((x, y)) # getpixel-Methode gibt die RGB-Werte des Pixels an der Position (x, y) zurück, z.B. (255, 0, 0) für Rot an der Pixelstelle (x, y)
        r2, g2, b2 = rgb_image2.getpixel((x, y))
        points1.append((r1, g1, b1, r1, g1, b1))  # Verwende RGB-Werte als Koordinaten und Farben für Bild 1
        points2.append((r2, g2, b2, r2, g2, b2))  # Verwende RGB-Werte als Koordinaten und Farben für Bild 2
        
# Konvertiere die Listen in Numpy-Arrays
points1 = np.array(points1)
points2 = np.array(points2)

# Schritt 7: Erstelle eine 3D-Figur und zeige die Punktewolken mit Farben an
fig = plt.figure(figsize=(12, 8))

# Subplot 1: 3D-Punktewolke für Bild 1
ax1 = fig.add_subplot(121, projection="3d") # 121: 1 Zeile, 2 Spalten, 1. Subplot
ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=points1[:, 3:6]/255, s=1) # scatter-Methode: Erstelle eine 3D-Punktewolke (Streudiagramm) mit den RGB-Werten als Koordinaten und Farben
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_zlabel("B")
ax1.set_xlim(0, 255)
ax1.set_ylim(0, 255)
ax1.set_zlim(0, 255)
ax1.set_title("Bild 1")

# Subplot 2: 3D-Punktewolke für Bild 2
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c=points2[:, 3:6]/255, s=1)
ax2.set_xlabel("R")
ax2.set_ylabel("G")
ax2.set_zlabel("B")
ax2.set_xlim(0, 255)
ax1.set_ylim(0, 255)
ax2.set_zlim(0, 255)
ax2.set_title("Bild 2")

# Schritt 8: Zeige die Ausgangsbilder als RGB-Bilder darunter an
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image1)
axes[0].set_title("Bild 1 (RGB)")
axes[0].axis("off")
axes[1].imshow(image2)
axes[1].set_title("Bild 2 (RGB)")
axes[1].axis("off")

# Schritt 9: Zeige die gesamte Darstellung an
plt.tight_layout() # tight_layout-Methode: Automatische Anpassung der Subplots, sodass sie nicht überlappen
plt.show()