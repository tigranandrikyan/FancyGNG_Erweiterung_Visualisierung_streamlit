### Run this script with the command:
# streamlit run src/streamlit_app_fancy_pca_quality_check.py ###

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import constants
import fancy_pca as FP
import math

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

            # Speichern der augmentierten Bild in der Liste
            all_images.append(aug_image)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}")  ### Debugging ###

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
