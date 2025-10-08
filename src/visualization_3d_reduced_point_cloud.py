### Run this visualization to get point cloud plots in 3D (RGB) 
# Colour points from the original image (left plot) and chosen colour values (right plot) in Figure 2 
# Here we use: images from 'data' as original, 'out_data' for FancyGNG and from 'out_data_pca' for FancyPCA ### 

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Bereite die Punkte in das richtige Format vor

image1_path = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/data/blumenwiese2.jpg"
image2_path = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/data/blumenwiese2.jpg"
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
        points1.append((r1, g1, b1, r1, g1, b1)) # Verwende RGB-Werte als Koordinaten und Farben für Bild 1
        points2.append((r2, g2, b2, r2, g2, b2)) # Verwende RGB-Werte als Koordinaten und Farben für Bild 2
        
# Konvertiere die Listen in Numpy-Arrays
points1 = np.array(points1)
points2 = np.array(points2)
### HINWEIS: Hier endet die Vorbereitung der Punkte in das richtige Format ###






# Erstelle eine 3D-Figur und zeige die Punktwolken mit Farben an


fig = plt.figure(figsize=(12, 8)) # Erstelle eine neue Matplotlib-Figur mit einer Größe von 12x8 Zoll

# Subplot 1: 3D-Punktewolke für Bild 1
ax1 = fig.add_subplot(121, projection="3d") # Erstelle ein 3D-Plot-Subplot auf der linken Seite (1. von 2)

# Erstelle eine 3D-Punktewolke mit den x-, y- und z-Koordinaten von `points1`
# Die Farben werden durch die RGB-Werte (Spalten 3-5) dargestellt (auf 0-1 normalisiert)
# `s=1` bedeutet, dass die Punkte sehr klein dargestellt werden
ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=points1[:, 3:6]/255, s=1)

# Achsenbeschriftungen setzen
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_zlabel("B")

# Begrenzungen für die Achsen setzen (Werte zwischen 0 und 255)
ax1.set_xlim(0, 255)
ax1.set_ylim(0, 255)
ax1.set_zlim(0, 255)

# Titel für das 3D-Diagramm setzen
ax1.set_title("Bild 1")

# enthält bestimmte vorgegebene RGB-Farben, die später dazu verwendet werden, die Farben der Punkte in der 3D-Punktewolke zu ersetzen -> Farben für 'Bild 2'
alt_colors_list = np.array([
    [189,30,91], 
    [25,60,216],
    [80,70,138], 
    [50,44,47], 
    [97,59,64], 
    [239,243,185], 
    [163,144,127], 
    [6,113,63]
])

print(points1[1,:])

# Funktion zur Bestimmung der ähnlichsten Farbe in `alt_colors_list`
def get_closest_color_2(pixel):
    distances = np.linalg.norm(alt_colors_list - pixel, axis=1) # Berechnet den euklidischen Abstand zu jeder Farbe in `alt_colors_list`
    return np.argmin(distances)  # Gibt nur den Index zurück und nicht wie bei 'get_closest_color' die Farbe selbst (siehe Zellen oben)

alt_colors = np.zeros([points1.shape[0],3]) # Initialisiere ein Array mit Nullwerten für alle Punkte in `points1` mit 3 Spalten (RGB)

# Farben ersetzen
for r in range(points1.shape[0]): # Iteriere durch alle Punkte in `points1`
    orig_color = points1[r,:3] # Extrahiere die ursprüngliche Farbe (RGB-Wert) des Punktes

    # Debugging: Falls der Index 'r' gleich 10 ist, drucke die Farbwerte aus
    if r==10:
        print(orig_color)
        print(points1[r,:3])
        
    ### Debugging-Code für Funktion 'get_closest_color_2' (nächsten zwei Zeilen auskommentieren): ###
    # closest_index = get_closest_color_2(orig_color)
    # print(f"orig_color: {orig_color}, closest_index: {closest_index}")
    alt_colors[r] = alt_colors_list[get_closest_color_2(orig_color)]


# Subplot 2: 3D-Punktewolke für Bild 2 (mit ersetzten Farben)
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=alt_colors/255, s=1)
ax2.set_xlabel("R")
ax2.set_ylabel("G")
ax2.set_zlabel("B")
ax2.set_xlim(0, 255)
ax2.set_ylim(0, 255)
ax2.set_zlim(0, 255)
ax2.set_title("Bild 2")

# fig.savefig("alt_cloud_orig.png")

# Zeige die Ausgangsbilder als RGB-Bilder darunter an
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image1)
axes[0].set_title("Bild 1 (RGB)")
axes[0].axis("off")
axes[1].imshow(image2)
axes[1].set_title("Bild 2 (RGB)")
axes[1].axis("off")

# Zeige die gesamte Darstellung an
plt.tight_layout()
plt.show()

