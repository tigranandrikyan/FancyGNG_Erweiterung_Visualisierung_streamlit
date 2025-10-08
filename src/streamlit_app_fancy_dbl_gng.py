import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
import matplotlib.pyplot as plt 

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

        # Anzeige der Cluster-Informationen
        st.subheader("Augmentierte Bilder:")

        # Spalten für das Layout, erstes Bild ist das Original, der Rest die augmentierten Bilder
        cols = st.columns(constants.AUG_COUNT + 1)  # Eine zusätzliche Spalte für das Originalbild -> veralteter Code

        # Originalbild anzeigen in der ersten Spalte
        with cols[0]:
            st.image(image, caption="Originalbild", use_container_width=True)

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

                # Anzeige der augmentierten Bilder in den entsprechenden Spalten
                with cols[aug_count + 1]:  # Beginne ab Spalte 1, weil Spalte 0 das Originalbild ist
                    st.image(aug_image, caption=f"Augmentation {aug_count + 1}", use_container_width=True)

            except Exception as e:
                st.write(f"Fehler bei der Augmentierung: {e}")  # Debugging

    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}") # Debugging