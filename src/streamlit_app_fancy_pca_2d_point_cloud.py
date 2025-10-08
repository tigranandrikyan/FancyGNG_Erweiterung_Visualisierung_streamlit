import streamlit as st
import numpy as np
import constants
import fancy_pca as FP
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit UI
st.title("Fancy PCA Image Augmentation")
st.write("Lade ein Bild hoch oder nimm eines mit der Kamera auf.")

# Option zur Bildaufnahme oder Datei-Upload
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

# Falls "Datei-Upload" gewählt wird, erscheint ein Dateiuploader
if input_option == "Datei-Upload":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
# Falls "Kamera" gewählt wird, erscheint eine Kameraaufnahme-Funktion
elif input_option == "Kamera":
    uploaded_file = st.camera_input("Bild aufnehmen")

if uploaded_file is not None:
    st.write(f"Dateityp: {type(uploaded_file)}")  ### Debugging: Zeigt den Dateityp an ###
    try:
        image = Image.open(uploaded_file)  # Öffnet das Bild mit PIL

        # Prüfe, ob RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}")  ### Debugging: Zeigt die Größe des Bildes an ###
    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}")  ### Debugging ###

    # Bild für Fancy PCA vorbereiten
    size_images = [image.size]  # Originalgröße speichern

    # Bild in ein NumPy-Array umwandeln und vertikal stapeln
    data_array = np.vstack(np.asarray(image)) / constants.MAX_COLOR_VALUE  # Normalisieren
    st.write(f"Bildarray Form: {data_array.shape}")  ### Debugging: Gibt die Form des Bildarrays aus ###
    data_list = [data_array]

    # Fancy PCA Transformation
    fancy_pca_transform = FP.FancyPCA()

    st.subheader("Augmentierte Bilder:")
    cols = st.columns(constants.AUG_COUNT)  # Spalten für die Anzeige der augmentierten Bilder -> nicht mehr nötig, ist ein Überbleibsel aus streamlit_app_fancy_pca.py

    # Liste, um das Originalbild und die augmentierten Bilder zu speichern
    all_images = [image]

    # Schleife zur Durchführung der Augmentierung für die Anzahl in AUG_COUNT
    for aug_count in range(constants.AUG_COUNT):
        try:
            # Wendet Fancy PCA auf das Bild an (Kopie wird verwendet, um Original nicht zu überschreiben, so wie bei fancy_pca_runner.py)
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array.copy())

            # Umwandlung auf den Bereich [0, 255] und zu uint8
            fancy_pca_images = (fancy_pca_images * 255).astype(np.uint8)

            # Rückumwandlung in die ursprüngliche Bildform: Höhe x Breite x 3
            height, width, channels = image.size[1], image.size[0], 3
            fancy_pca_images = fancy_pca_images.reshape((height, width, channels))

            st.write(f"Form des transformierten Bildes: {fancy_pca_images.shape}")  # Debugging: Gibt die Form des transformierten Bildes aus
        except Exception as e:
            st.write(f"Fehler bei Fancy PCA: {e}")  ### Debugging ###

        # Konvertiert das NumPy-Array zurück in ein Bildformat zur Anzeige
        try:
            aug_image = Image.fromarray(fancy_pca_images)

            # Das augmentierte Bild in der Liste speichern
            all_images.append(aug_image)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}")  ### Debugging ###

    # Visualisierung der 2D-Punktewolke für das Originalbild und die augmentierten Bilder
    st.subheader("Visualisierung der 2D-Punktwolke")

    fig, axs = plt.subplots(2, len(all_images), figsize=(15, 6))

    # Originalbild & 2D-Punktewolke (PCA oder Fancy PCA) für das Originalbild
    # Konvertiere das Originalbild in eine 2D-Punktwolke
    rgb_image = all_images[0].convert("RGB")
    width, height = all_images[0].size
    points = []
    for x in range(width):
        for y in range(height):
            r, g, b = rgb_image.getpixel((x, y))
            points.append((r, g, b, r, g, b))  # RGB-Werte als Koordinaten und Farben verwenden
    points = np.array(points)
    # points[:, 1]: zweite Spalte (x-Achse-Werte - Grün), points[:, 2]: dritte Spalte (y-Achse-Werte - Blau), c=points[:, 3:6] / 255: Farbwerte auf [0,1] normalisiert
    axs[0, 0].scatter(points[:, 1], points[:, 2], c=points[:, 3:6] / 255, s=1)
    axs[0, 0].set_title("Originalbild")
    axs[0, 0].set_xlabel("G")
    axs[0, 0].set_ylabel("B")
    axs[0, 0].set_xlim(0, 255)
    axs[0, 0].set_ylim(0, 255)
    axs[0, 0].set_aspect('equal', 'box')
    axs[1, 0].imshow(all_images[0])
    axs[1, 0].axis("off")

    # Augmentierte Bilder & 2D-Punktwolke für jedes augmentierte Bild
    for idx, aug_image in enumerate(all_images[1:], start=1):
        rgb_image = aug_image.convert("RGB")
        width, height = aug_image.size
        points = []
        for x in range(width):
            for y in range(height):
                r, g, b = rgb_image.getpixel((x, y))
                points.append((r, g, b, r, g, b))  # RGB-Werte als Koordinaten und Farben verwenden
        points = np.array(points)

        axs[0, idx].scatter(points[:, 1], points[:, 2], c=points[:, 3:6] / 255, s=1)
        axs[0, idx].set_title(f"Augmentation {idx}")
        axs[0, idx].set_xlabel("G")
        axs[0, idx].set_ylabel("B")
        axs[0, idx].set_xlim(0, 255)
        axs[0, idx].set_ylim(0, 255)
        axs[0, idx].set_aspect('equal', 'box')
        axs[1, idx].imshow(aug_image)
        axs[1, idx].axis("off")

    plt.tight_layout()
    st.pyplot(fig)