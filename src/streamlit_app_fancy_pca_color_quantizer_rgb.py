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
    st.write(f"Dateityp: {type(uploaded_file)}")  # Debugging

    try:
        image = Image.open(uploaded_file)  # Öffnet das Bild mit PIL

        # Prüfe, ob RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}")  # Debugging
    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}")  # Debugging

    # Bild für Fancy PCA vorbereiten
    size_images = [image.size]  # Originalgröße speichern

    # Bild in ein NumPy-Array umwandeln und normalisieren
    data_array = np.vstack(np.asarray(image)) / constants.MAX_COLOR_VALUE
    st.write(f"Bildarray Form: {data_array.shape}")  # Debugging
    data_list = [data_array]

    # Fancy PCA Transformation
    fancy_pca_transform = FP.FancyPCA()

    # Augmentierung durchführen
    all_images = [image]

    for aug_count in range(constants.AUG_COUNT):
        try:
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array.copy())
            fancy_pca_images = (fancy_pca_images * 255).astype(np.uint8)
            height, width, channels = image.size[1], image.size[0], 3
            fancy_pca_images = fancy_pca_images.reshape((height, width, channels))

            st.write(f"Form des transformierten Bildes: {fancy_pca_images.shape}")  # Debugging
        except Exception as e:
            st.write(f"Fehler bei Fancy PCA: {e}")  # Debugging

        # Konvertiert das NumPy-Array zurück in ein Bild
        try:
            aug_image = Image.fromarray(fancy_pca_images)
            all_images.append(aug_image)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}")  # Debugging

    st.subheader('Farbquantisierung der Bilder')

    # Funktion zur Bestimmung der ähnlichsten Farbe
    def get_closest_color(pixel, colors):
        distances = np.linalg.norm(colors - pixel, axis=1)  # berechne entlang der Zeilen (axis=1) die euklidische Norm von 'colors-pixel'
        closest_color_index = np.argmin(distances)  # gibt den Index des kleinsten Elements zurück
        return closest_color_index

    # Vorgegebene Farben
    colors = np.array([
        [189, 30, 91],
        [25, 60, 216],
        [80, 70, 138],
        [50, 44, 47],
        [97, 59, 64],
        [239, 243, 185],
        [163, 144, 127],
        [6, 113, 63]
    ])

    # Farben quantisieren (ersetzen) für jedes Bild
    def quantize_image(image, colors):
        pixels = np.array(image)
        for row in range(pixels.shape[0]):
            for col in range(pixels.shape[1]):
                pixel = pixels[row, col]
                closest_color_index = get_closest_color(pixel, colors)
                pixels[row, col] = colors[closest_color_index]
        return Image.fromarray(pixels)

    # Zeige das Originalbild und augmentierte Bilder zusammen mit ihren quantisierten Versionen
    fig, axes = plt.subplots(2, len(all_images), figsize=(15, 6))

    for idx, img in enumerate(all_images):
        # Zeige das Originalbild und augmentierte Bilder
        axes[0, idx].imshow(img)
        if idx == 0:
            axes[0, idx].set_title("Originalbild")
        else:
            axes[0, idx].set_title(f"Augmentiertes Bild {idx}")

        axes[0, idx].axis("off")

        # Zeige das quantisierte Bild
        quantized_img = quantize_image(img, colors)
        axes[1, idx].imshow(quantized_img)
        if idx == 0:
            axes[1, idx].set_title("Originalbild (quantisiert)")
        else:
            axes[1, idx].set_title(f"Aug. Bild {idx} (quantisiert)")
        axes[1, idx].axis("off")

    plt.tight_layout()
    st.pyplot(fig)