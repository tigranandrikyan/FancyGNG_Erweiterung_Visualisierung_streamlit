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

    st.subheader("Augmentierte Bilder:")
    cols = st.columns(constants.AUG_COUNT)  # Spalten für Anzeige der augmentierten Bilder

    # Liste für Originalbild und augmentierte Bilder
    all_images = [image]

    # Augmentierung durchführen
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

    # Visualisierung der RGB 3D-Punktwolken
    st.subheader("3D-RGB-Punktwolke Visualisierung")

    fig = plt.figure(figsize=(12, 8))

    for idx, img in enumerate(all_images):
        rgb_image = img.convert("RGB")
        width, height = img.size
        points = []

        for x in range(width):
            for y in range(height):
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

    # Tight Layout ist eine gute Möglichkeit, um automatisch Überlappungen zu vermeiden
    plt.tight_layout()

    # Verwende 'subplots_adjust', um den Abstand explizit anzupassen, falls nötig
    fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Wspace für horizontalen Abstand, Hspace für vertikalen Abstand

    # Ausgabe der Visualisierung
    st.pyplot(fig)

    # Originalbild und augmentierte Bilder anzeigen
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