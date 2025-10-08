import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import math 

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

        # Berechnung und Visualisierung der Metriken
        def calculate_metrics(image1, image2):
            mse_value = mean_squared_error(image1.flatten(), image2.flatten())
            
            # Berechnung von SSIM - guter Wert liegt bei > 0.9 -> visuelle Struktur und das allgemeine Erscheinungsbild des Bildes gut erhalten geblieben
            try:
                ssim_value = ssim(image1, image2, multichannel=True, win_size=3, channel_axis=2)
            except ValueError:
                ssim_value = None
            
            # Berechnung von PSNR - guter Wert liegt bei > 30 dB -> bei < 30 dB: es gibt signifikante pixelgenaue Unterschiede zwischen den Bildern
            psnr_value = calculate_psnr(image1, image2)
            
            return mse_value, ssim_value, psnr_value

        # Funktion zur Berechnung von PSNR
        def calculate_psnr(image1, image2):
            mse_value = mean_squared_error(image1.flatten(), image2.flatten())
            if mse_value == 0:
                return float('inf')  # Kein Unterschied
            max_pixel = 255.0
            psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse_value))
            return psnr_value

        # Originalbild (Bild 1)
        image1 = np.array(all_images[0])

        # Berechnen der Metriken für jedes augmentierte Bild
        mse_values = []
        ssim_values = []
        psnr_values = []
        for i, aug_image in enumerate(all_images[1:], start=1):
            aug_image_array = np.array(aug_image)
            mse_value, ssim_value, psnr_value = calculate_metrics(image1, aug_image_array)
            mse_values.append(mse_value)
            ssim_values.append(ssim_value)
            psnr_values.append(psnr_value)

        # Zeige nur die letzte Visualisierung mit allen Bildern
        st.subheader("Visualisierung der Bilder mit Metriken:")

        # Plot der Metriken und Bilder
        fig, axes = plt.subplots(1, len(all_images), figsize=(18, 6))

        # Bild 1 (Original) anzeigen
        axes[0].imshow(image1)
        axes[0].set_title("Originalbild")
        axes[0].axis("off")

        for i, aug_image in enumerate(all_images[1:], start=1):
            aug_image_array = np.array(aug_image)
            mse_value = mse_values[i-1]
            ssim_value = ssim_values[i-1]
            psnr_value = psnr_values[i-1]

            axes[i].imshow(aug_image)
            axes[i].set_title(f"Augmentiertes Bild {i}")
            axes[i].axis("off")
            axes[i].text(0.5, -0.2, f"MSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}\nPSNR: {psnr_value:.4f}", ha='center', va='center', transform=axes[i].transAxes)

        # Layout anpassen und Bilder anzeigen
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}") # Debugging