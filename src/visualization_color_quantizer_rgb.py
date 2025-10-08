### Run this visualization to use only eight RGB colours (predefined) for the nine plots by the next closest colour 
# first picture = original image, last picture = fully quantized, the other ones are intermediate steps
# note: if the 'else' code block in the optional part (two lines) will be commented out, there will be used (partly) grayscales ###

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Bild einlesen
image_path = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/data/blumenwiese2.jpg"
image = Image.open(image_path)

# Vorgegebene Farben
#colors = np.array([
#    [255, 0, 0],     # Rot
#    [0, 255, 0],     # Grün
#    [0, 0, 255],     # Blau
#    [255, 255, 0],   # Gelb
#    [255, 0, 255],   # Magenta
#    [0, 255, 255],   # Cyan
#    [128, 128, 128], # Grau
#    [0, 0, 0]        # Schwarz
#])

colors = np.array([
    [189,30,91], 
    [25,60,216],
    [80,70,138], 
    [50,44,47], 
    [97,59,64], 
    [239,243,185], 
    [163,144,127], 
    [6,113,63]
])

# Funktion um RGB-Werte in Grauwerte umzuwandeln 
def rgb_to_gray(rgb): # rgb: Array mit 3 Elementen (r,g,b)
    # Gewichtungsfaktoren für die RGB-Kanäle
    r_factor = 0.2989
    g_factor = 0.5870
    b_factor = 0.1140
    
    # Grauwert berechnen
    #gray = np.min((3.4*((rgb[0] * r_factor) + (rgb[1] * g_factor) + (rgb[2] * b_factor)),255))
    gray = ((rgb[0] * r_factor) + (rgb[1] * g_factor) + (rgb[2] * b_factor))
    
    return int(gray) # man braucht nur einen Wert für grau zurückgeben, da grau = r=g=b

# Funktion zur Bestimmung der ähnlichsten Farbe
def get_closest_color(pixel):
    distances = np.linalg.norm(colors - pixel, axis=1) # berechne entlang der Zeilen (axis=1) die euklidische Norm von 'colors-pixel'
    closest_color_index = np.argmin(distances) # gibt den Index des kleinsten Elements zurück
    return closest_color_index

# Farben in Grautöne umwandeln
gray_colors = np.array(colors) # Kopie der Farben
for k in range(gray_colors.shape[0]): # shape[0] gibt die Anzahl der Zeilen zurück
    gray_colors[k] = rgb_to_gray(colors[k]) # Grauwert für jede Zeile (Farbe) berechnen
    
colors_hist = np.zeros(colors.shape[0], dtype=int) # erstelle ein Array mit Nullen (der Datentyp 'Integer' soll verwendet werden -> dtype=int), das die Anzahl der Farben enthält

print(gray_colors)
print(colors_hist)

# Serie von Bildern erzeugen
num_images = colors.shape[0]+1 # Anzahl der Farben + 1 (für das Originalbild)
output_images = []
output_images.append(image.copy()) # Originalbild zur Ausgabeliste hinzufügen

for i in range(num_images):
    # Bild kopieren und Pixelwerte abrufen
    output_image = image.copy() # Kopie des Originalbildes
    pixels = np.array(output_image) # Pixelwerte des Bildes in ein Array umwandeln
    colors_hist = np.zeros(colors.shape[0], dtype=int) # erstelle ein Array mit Nullen (der Datentyp 'Integer' soll verwendet werden -> dtype=int), das die Anzahl der Farben enthält
    
    # Farben ersetzen
    for row in range(pixels.shape[0]): # shape[0] gibt die Anzahl der Zeilen zurück
        for col in range(pixels.shape[1]): # shape[1] gibt die Anzahl der Spalten zurück
            pixel = pixels[row, col] # Pixelwert an der Position (row, col) abrufen
            closest_color_index = get_closest_color(pixel) # Index der ähnlichsten Farbe bestimmen
            
            # Schrittweise Quantisierung der Bilder
            if closest_color_index<=i: # wenn der Index der Farbe kleiner oder gleich dem Index i ist, dann... (d.h., wir haben alle Farben bis zur i-ten Farbe berücksichtigt für das i-te Bild)
                pixels[row, col] = colors[closest_color_index] # ersetze den Pixelwert durch den Wert der Farbe
                colors_hist[closest_color_index]+=1 # erhöhe den Zähler der verwendeten Farbe um 1, d.h. wie oft die Farbe verwendet wurde, um die Häufigkeit der Farben zu tracken
            
            ### Optional: Färbe die Farben in Graustufen ein (auskommentieren der beiden folgenden Zeilen, d.h. else-Block) ###
            #else:
                #pixels[row, col] = gray_colors[closest_color_index]
    
    # Bearbeitetes Bild zur Ausgabeliste hinzufügen
    output_images.append(Image.fromarray(pixels)) # Image.fromarray(pixels): Erstelle ein Bild (PNG oder JPEG) aus einem Array durch die PIL-Bibliothek
    #output_images[i].save("mohnfeld{0}.png".format(i))
    print(colors_hist) # Ausgabe der getrackten Farbhäufigkeiten

# Bilder anzeigen
plt.figure(figsize=(10, 10))
for i in range(num_images):
    #output_images[i].save("mohnfeld{0}.png".format(i)) # speichere die Bilder ({0} wird durch i ersetzt -> String-Formatierung)
    plt.subplot(3, 3, i + 1) # 3 Zeilen, 3 Spalten, i+1: Index des Subplots
    plt.imshow(output_images[i])
    plt.axis('off')

plt.show()
print(colors_hist) # Ausgabe der getrackten Farbhäufigkeiten
