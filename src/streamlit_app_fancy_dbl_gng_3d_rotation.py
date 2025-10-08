import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
import matplotlib.pyplot as plt
import random
from io import BytesIO
import imageio.v2 as imageio
import os

# Konstanten initialisieren
MAX_PIXELS = 10000  # Maximale Anzahl von Pixeln für die Bildverkleinerung
SAMPLE_SIZE_FACTOR = 0.1  # Faktor für die Auswahl von 10% der Pixel (z.B. 0.1 für 10%)

# Funktion zum Verkleinern des Bildes, falls zu viele Pixel vorhanden sind (MAX_PIXELS)
def resize_image(image, max_pixels=MAX_PIXELS):
    width, height = image.size
    total_pixels = width * height
    if total_pixels > max_pixels:
        scale_factor = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        st.write(f"Bild verkleinert auf: {new_width}x{new_height}")  # Debugging
    return image

# Streamlit UI
st.title("DBL-GNG Image Augmentation")
st.write("Lade ein Bild hoch oder nimm eines mit der Kamera auf.")

# Option zur Bildaufnahme oder Datei-Upload
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

if input_option == "Datei-Upload":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
elif input_option == "Kamera":
    uploaded_file = st.camera_input("Bild aufnehmen")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        # Überprüfen, ob das Bild im richtigen Modus ist (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Bildgröße anzeigen
        st.write(f"Bildgröße: {image.size}")

        # Bild auf maximale Pixelzahl verkleinern, falls notwendig
        image = resize_image(image)

        # Bild in ein NumPy-Array umwandeln und sicherstellen, dass es die richtige Form hat (n_pixels, n_features)
        image_array = np.asarray(image)
        data_array = image_array.reshape(-1, 3) / constants.MAX_COLOR_VALUE  # Normalisiere die Daten zur Weiterverarbeitung (wie in parser.py); reshape besser als vstack

        st.write(f"Bildarray Form: {data_array.shape}") ### Debugging: Gibt die Form des Bildarrays aus ###

        # Initialisierung von DBL-GNG für die Augmentierung
        gng = dbl_gng.DBL_GNG(3, constants.MAX_NODES)  # GNG mit 3 Dimensionen und MAX_NODES
        gng.initializeDistributedNode(data_array, constants.SARTING_NODES)  # Initialisiere verteilte Knoten mit den Bilddaten

        # Training des GNG-Modells
        bar = trange(constants.EPOCH)  # Fortschrittsbalken für Trainingsepochen
        for i in bar:
            gng.resetBatch()  # Zurücksetzen des Batch-Trainings
            gng.batchLearning(data_array)  # Batch-Lernen
            gng.updateNetwork()  # Netzwerk aktualisieren
            gng.addNewNode(gng)  # Neuen Knoten hinzufügen
            bar.set_description(f"Epoch {i + 1} Knotenanzahl: {len(gng.W)}")  # Fortschrittsanzeige

        gng.cutEdge()  # Entferne nicht benötigte Kanten im Netzwerk
        gng.finalNodeDatumMap(data_array)  # Finales Mapping der Knoten zu den Datenpunkten

        # Abrufen der finalen Distanzmatrix und Knoten
        finalDistMap = gng.finalDistMap
        finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int)  # Berechne finale Knoten und skaliere
        connectiveMatrix = gng.C  # Verbindungs-Matrix zwischen Knoten
        pixel_cluster_map, node_cluster_map = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix)
        pixel_cluster_map = np.array(pixel_cluster_map)  # Clustering durchführen

        cluster_count = int(max(node_cluster_map)) + 1  # Berechnung der Anzahl der Cluster; +1, weil Cluster bei 0 beginnt

        # Anzeige der Cluster-Informationen
        st.write(f"Anzahl der Cluster: {cluster_count}")

        # Spalten für das Layout, erstes Bild ist das Original, der Rest die augmentierten Bilder
        cols = st.columns(constants.AUG_COUNT + 1)  # Eine zusätzliche Spalte für das Originalbild -> veralteter Code

        all_images = [image]

        # Augmentierungen durchführen
        for aug_count in trange(constants.AUG_COUNT): # Fortschrittsbalken für Augmentierungen
            try:
                # Augmentierung der Daten mit den Clusterfarben 
                # [image.size] ist das gleiche wie size_images aus parser.py; data_array ist das gleiche wie data[data_index] aus fancy_pca/dbl_gng_runner.py
                aug_data = color_pca.modify_clusters(data_array, pixel_cluster_map, cluster_count, [image.size], 0)
                aug_data = (aug_data * 255).astype(np.uint8)  # Umwandlung in uint8

                # Rückwandlung in die ursprüngliche Bildform: Höhe x Breite x 3
                aug_data = aug_data.reshape((image.size[1], image.size[0], 3))

                # Erstellen des augmentierten Bildes
                aug_image = Image.fromarray(aug_data)
                all_images.append(aug_image)

            except Exception as e:
                st.write(f"Fehler bei der Augmentierung: {e}")  # Debugging

        # Visualisierung der 3D-Rotationsanimation
        st.subheader("3D RGB Rotations-Animation")  # Titel für das GIF

        fig = plt.figure(figsize=(12, 8)) # Diese Größe nötig, damit Schrift erkennbar bleibt

        for idx, img in enumerate(all_images):
            rgb_image = img.convert("RGB")
            width, height = img.size
            points = []

            # Sampling der Pixel gemäß SAMPLE_SIZE_FACTOR
            sample_size = int(SAMPLE_SIZE_FACTOR * width * height)
            sampled_pixels = random.sample(range(width * height), sample_size)

            # Der Code durchläuft eine Liste von zufällig ausgewählten Pixelindizes (sampled_pixels) aus einem Bild 
            # und extrahiert die RGB-Farbwerte der entsprechenden Pixel. Diese Farbwerte werden dann in eine Liste namens points gespeichert
            for i in sampled_pixels:
                x = i % width # gibt die x-Koordinate (Spalte) zurück, z.B.: x = 7 % 5 = 2 (3. Spalte, da x=2) bei i=7 und width=5
                y = i // width # gibt die x-Koordinate (Spalte) zurück, z.B.: y = 7 // 5 = 1 (2. Zeile, da y=1) bei i=7 und width=5
                r, g, b = rgb_image.getpixel((x, y))
                points.append((r, g, b))  # Nur RGB-Werte verwenden

            points = np.array(points)

            ax = fig.add_subplot(1, len(all_images), idx + 1, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points / 255, s=1)  # Punktgröße vergrößern
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")

            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_zlim(0, 255)

            # Setze den richtigen Titel
            if idx == 0:
                ax.set_title("Originalbild")
            else:
                ax.set_title(f"Augmentiertes Bild {idx}")

        plt.tight_layout() # Optimiert automatisch den Abstand

        # Verwende 'subplots_adjust', um den Abstand explizit anzupassen, falls nötig
        #fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Wspace für horizontalen Abstand, Hspace für vertikalen Abstand

        # 3D-Rotationsanimation
        num_frames = 40
        angles = np.linspace(0, 360, num_frames) # hat 'num_frames' Werte zwischen 0 und 360
        frames = []

        for angle in angles:
            for ax in fig.get_axes(): # geht durch alle 3D-Plots und aktualisiert alle 3D-Plots
                ax.view_init(30, angle)  # Kamera-Rotation; 30: Höhe, angle: Drehung
            plt.draw()

            # Bild direkt aus dem Speicher speichern
            buf = BytesIO()
            plt.savefig(buf, format='png') 
            buf.seek(0)
            frames.append(imageio.imread(buf))

        # Definiere Zielverzeichnis, wo GIF gespeichert wird (WICHTIG: Speichern ist nötig, damit streamlit das GIF anzeigen kann; Byte-Code reicht nicht immer für stabile Darstellung aus)
        gif_dir = os.path.dirname(os.path.abspath(__file__))  # Verzeichnis des aktuellen Skripts, was für alle Nutzer funktioniert
        #gif_dir = "/Users/macbookair/Desktop/Materialien/Python/FancyGNG_Erweiterung_Visualisierung_streamlit/" # meine lokale Speicheradresse
        gif_path = os.path.join(gif_dir, "3d_rotation_animation_gng.gif")

        # GIF im Zielverzeichnis speichern
        imageio.mimsave(gif_path, frames, duration=0.05, loop=0)  # Setze "loop=0" für eine Endlosschleife

        # GIF in Streamlit anzeigen
        st.image(gif_path, caption="3D RGB Rotations-Animation", use_container_width=True)

        # Augmentierte Bilder unter dem GIF anzeigen
        st.subheader("Augmentierte Bilder")
        fig, axes = plt.subplots(1, len(all_images), figsize=(12, 6))
        for idx, img in enumerate(all_images):
            axes[idx].imshow(img)

            # Setze den richtigen Titel
            if idx == 0:
                axes[idx].set_title("Originalbild")
            else:
                axes[idx].set_title(f"Augmentiertes Bild {idx}")

            axes[idx].axis("off")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}") # Debugging