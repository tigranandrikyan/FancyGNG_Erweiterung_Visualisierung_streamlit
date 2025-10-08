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

    # -------------------------- Funktionen zur Bildverarbeitung --------------------------

    # Visualisierung der RGB 3D-Punktwolken
    st.subheader("Reduzierte 3D-RGB-Punktwolke Visualisierung")

    # Farbliste zum Ersetzen
    alt_colors_list = np.array([
        [189, 30, 91], 
        [25, 60, 216],
        [80, 70, 138], 
        [50, 44, 47], 
        [97, 59, 64], 
        [239, 243, 185], 
        [163, 144, 127], 
        [6, 113, 63]
    ])

    def get_closest_color_2(pixel):
        distances = np.linalg.norm(alt_colors_list - pixel, axis=1)
        return np.argmin(distances)

    def replace_colors_with_alt(image):
        image_np = np.array(image)
        reshaped = image_np.reshape(-1, 3)
        replaced = np.array([
            alt_colors_list[get_closest_color_2(pixel)] for pixel in reshaped
        ])
        replaced_image = replaced.reshape(image_np.shape).astype(np.uint8)
        return Image.fromarray(replaced_image)

    def plot_3d_pointcloud(image, ax, title):
        rgb_image = image.convert("RGB")
        width, height = image.size
        points = []

        for x in range(width):
            for y in range(height):
                r, g, b = rgb_image.getpixel((x, y))
                points.append((r, g, b))

        points = np.array(points)

        alt_colors = np.zeros([points.shape[0], 3])
        for r in range(points.shape[0]):
            orig_color = points[r, :3]
            alt_colors[r] = alt_colors_list[get_closest_color_2(orig_color)]

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=alt_colors / 255, s=1)
        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.set_title(title)

    # -------------------------- Visualisierung --------------------------

    # Original 3D Punktwolke + Augmentierte
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, len(all_images), 1, projection="3d")
    plot_3d_pointcloud(all_images[0], ax1, "Originalbild")
    for idx, aug_image in enumerate(all_images[1:], 2):
        ax = fig.add_subplot(1, len(all_images), idx, projection="3d")
        plot_3d_pointcloud(aug_image, ax, f"Augmentiertes Bild {idx-1}")
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    st.pyplot(fig)

    # Anzeige der Original- und augmentierten Bilder
    fig, axes = plt.subplots(1, len(all_images), figsize=(12, 6))
    for idx, img in enumerate(all_images):
        axes[idx].imshow(img)
        axes[idx].set_title("Originalbild" if idx == 0 else f"Augmentiertes Bild {idx}")
        axes[idx].axis("off")
    plt.tight_layout()
    st.pyplot(fig)

    # Farben ersetzen mit alt_colors_list
    replaced_images = [replace_colors_with_alt(img) for img in all_images]

    # Anzeige der farbersetzten Bilder
    fig, axes = plt.subplots(1, len(replaced_images), figsize=(12, 6))
    for idx, img in enumerate(replaced_images):
        axes[idx].imshow(img)
        axes[idx].set_title("Ersetzt: Original" if idx == 0 else f"Ersetzt: Aug. Bild {idx}")
        axes[idx].axis("off")
    plt.tight_layout()
    st.pyplot(fig)