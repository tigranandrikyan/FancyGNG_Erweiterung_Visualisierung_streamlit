from PIL import Image
import numpy as np
import glob
import constants
import os
from PIL import Image
import shutil
from datetime import datetime

#Vorbereitung der Daten 

# Liest Bilder ein, konvertiert sie, normalisiert sie und speichert ihre Größe
def parse(file_list): # Definiert die Funktion parse, die eine Liste von Bilddateien (file_list) als Eingabe erhält
    data_list = [] # Speichert die Bilddaten als normalisierte Arrays
    size_images = [] # Speichert die Größe der Bilder
    for image_file in file_list: # Startet eine Schleife, die jedes Bild in file_list verarbeitet
        try: # Versucht, den folgenden Codeblock auszuführen, um Fehler abzufangen
            image = Image.open(image_file) # Öffnet die Bilddatei
            if image.mode != 'RGB': # Überprüft, ob das Bild nicht bereits im RGB-Modus ist
                    image = image.convert('RGB') # Falls das Bild nicht im RGB-Format ist, wird es entsprechend konvertiert
            size_images.append(image.size) # Speichert die Bildgröße (Breite, Höhe) in size_images
            data_array = np.vstack(np.asarray(image)) # Konvertiert das Bild in ein NumPy-Array und stapelt zeilenweise (vertikale Verkettung)
            data_list.append(data_array/constants.MAX_COLOR_VALUE) # Normalisiert das Bild, indem es durch MAX_COLOR_VALUE geteilt wird, und speichert es in data_list
        except Exception as e: # Fehler abfangen
            print(f"Error processing image {image_file}: {str(e)}")
    return data_list, size_images # Gibt die Liste der normalisierten Bilddaten (data_list) und die Liste der Bildgrößen (size_images) zurück

# Erstellt eine Liste aller Bilddateien in einem Verzeichnis
def generate_file_list(image_in_path = './data'): # Definiert die Funktion generate_file_list, mit einem Standardpfad ./data

    # glob: durchsucht Dateisystem nach Dateien; image_in_path: Basisordner='./data'; '/*': suche nach allen Dateien im Verzeichnis; FILE_TYPE: Dateiendung aus constants.py
    file_list = glob.glob(image_in_path + '/*' + constants.FILE_TYPE)

    return file_list # Gibt die Liste der gefundenen Bilddateien zurück

# Extrahiert die Bildnamen ohne Endung
def get_image_names(file_list): # Erhält eine Liste von Dateipfaden (file_list)
    name_list = [] # zum Speichern von Bildnamen
    for file_path in file_list: # Startet eine Schleife, die jeden Dateipfad in file_list verarbeitet
       image = os.path.basename(file_path) # Extrahiert den Dateinamen (mit Endung) aus dem Pfad
       image_name = os.path.splitext(image)[0] # Trennt die Dateiendung ab und speichert nur den reinen Namen; [0]: Zugriff auf den Dateinamen ohne Datei-Endung
       name_list.append(image_name) # Fügt den Bildnamen der name_list hinzu
    return name_list # Rückgabe der Liste der Bildnamen

### TODO: ggf. meinen Code zur Augmentierung (auch) hier einfügen/anpassen ###

# Speichert augmentierte Bilder in einem neuen Verzeichnis
def save_data(aug_data, name, size_images, aug_count, data_index, path='./out_data'): # path='./out_data': Speicherort für das Bild (Standard: './out_data')
    aug_image = aug_data * constants.MAX_COLOR_VALUE # Skaliert die normalisierten Pixelwerte zurück auf den ursprünglichen Bereich durch MAX_COLOR_VALUE

    # Wandelt das 1D-Array zurück in die ursprüngliche Bildgröße und -form (Höhe x Breite x 3) und konvertiert es in uint8, das für Bilder erforderlich ist
    aug_image = aug_image.reshape((size_images[data_index][1], size_images[data_index][0], 3)).astype(np.uint8) # [1]: Höhe; [0]: Breite; 3: RGB

    image = Image.fromarray(aug_image) # Erstellt ein PIL-Bild aus dem NumPy-Array
    final_image = image.convert("RGB") # Konvertiert das Bild erneut in den RGB-Modus (falls nötig)

    # Speichert das Bild unter dem angegebenen Pfad mit einer durchnummerierten Erweiterung (name_1, name_2, ...) und Datei-Endung 'FILE_TYPE' aus constants.py
    # Windows:
    #final_image.save(path + '\\' + name + "_" + str(aug_count + 1) + constants.FILE_TYPE)
    # Mac/Linux:
    # final_image.save(path + '/' + name + "_" + str(aug_count + 1) + constants.FILE_TYPE)

    ### HINWEIS: in Windows und Mac/Linux werden die Backslashes verschieden genutzt, daher: 
    # plattformübergreifende Lösung für Windows/Mac/Linux (zwei Zeilen statt eine Zeile):

    # Windows, Mac/Linux:
    save_path = os.path.join(path, name + "_" + str(aug_count + 1) + constants.FILE_TYPE)
    final_image.save(save_path)
    
    # statt nur (suboptimal): final_image.save(path + '\\' + name + "_" + str(aug_count + 1) + constants.FILE_TYPE) ###

