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
    st.write(f"Dateityp: {type(uploaded_file)}")  ### Debugging ###
    try:
        image = Image.open(uploaded_file)

        # Prüfe, ob RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}")  ### Debugging ###
    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}")  ### Debugging ###

    # Bild für Fancy PCA vorbereiten
    size_images = [image.size]  

    # Bild in ein NumPy-Array umwandeln und normalisieren
    data_array = np.vstack(np.asarray(image)) / constants.MAX_COLOR_VALUE  
    st.write(f"Bildarray Form: {data_array.shape}")  ### Debugging ###
    data_list = [data_array]

    # Fancy PCA Transformation
    fancy_pca_transform = FP.FancyPCA()

    st.subheader("Augmentierte Bilder:")
    cols = st.columns(constants.AUG_COUNT)  

    # Liste, um das Originalbild und die augmentierten Bilder zu speichern
    all_images = [image]

    # Schleife zur Durchführung der Augmentierung
    for aug_count in range(constants.AUG_COUNT):
        try:
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array.copy())

            # Umwandlung auf [0, 255] und zu uint8
            fancy_pca_images = (fancy_pca_images * 255).astype(np.uint8)

            # Rückumwandlung in Höhe x Breite x 3
            height, width, channels = image.size[1], image.size[0], 3
            fancy_pca_images = fancy_pca_images.reshape((height, width, channels))

            st.write(f"Form des transformierten Bildes: {fancy_pca_images.shape}")  
        except Exception as e:
            st.write(f"Fehler bei Fancy PCA: {e}")  ### Debugging ###

        try:
            aug_image = Image.fromarray(fancy_pca_images)
            all_images.append(aug_image)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}")  ### Debugging ###

    # Visualisierung der RGB-Farbverteilung
    st.subheader("RGB-Farbverteilung für Original- und Augmentierte Bilder")

    for idx, img in enumerate(all_images):
        # Generiere den Bildnamen basierend auf dem Index
        if idx == 0:
            bild_name = "Originalbild"
        else:
            bild_name = f"Augmentiertes Bild {idx}"

        # RGB-Werte extrahieren
        width, height = img.size
        r_values, g_values, b_values, colors = [], [], [], []

        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                r_values.append(r)
                g_values.append(g)
                b_values.append(b)
                colors.append((r / 255, g / 255, b / 255))  

        r_values = np.array(r_values)
        g_values = np.array(g_values)
        b_values = np.array(b_values)

        # Plots erstellen
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].scatter(r_values, g_values, c=colors, s=1)
        axes[0, 0].set_xlabel("R")
        axes[0, 0].set_ylabel("G")
        axes[0, 0].set_xlim(0, 255)
        axes[0, 0].set_ylim(0, 255)
        axes[0, 0].set_title("RG-Farbverteilung")

        axes[0, 1].scatter(r_values, b_values, c=colors, s=1)
        axes[0, 1].set_xlabel("R")
        axes[0, 1].set_ylabel("B")
        axes[0, 1].set_xlim(0, 255)
        axes[0, 1].set_ylim(0, 255)
        axes[0, 1].set_title("RB-Farbverteilung")

        axes[1, 1].scatter(g_values, b_values, c=colors, s=1)
        axes[1, 1].set_xlabel("G")
        axes[1, 1].set_ylabel("B")
        axes[1, 1].set_xlim(0, 255)
        axes[1, 1].set_ylim(0, 255)
        axes[1, 1].set_title("GB-Farbverteilung")

        axes[1, 0].imshow(img)
        axes[1, 0].set_title(bild_name)  # Bildname verwenden
        axes[1, 0].axis("off")

        plt.tight_layout()
        st.pyplot(fig)