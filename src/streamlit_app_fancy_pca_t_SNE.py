import streamlit as st
import numpy as np
import constants
import fancy_pca as FP
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Streamlit UI
st.title("Fancy PCA Image Augmentation & t-SNE Visualisierung")
st.write("Lade ein Bild hoch oder nimm eines mit der Kamera auf.")

# Option zur Bildaufnahme oder Datei-Upload
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

if input_option == "Datei-Upload":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
elif input_option == "Kamera":
    uploaded_file = st.camera_input("Bild aufnehmen")

if uploaded_file is not None:
    st.write(f"Dateityp: {type(uploaded_file)}")
    try:
        image = Image.open(uploaded_file)

        # Prüfe, ob RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}")
    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}")

    # Bild für Fancy PCA vorbereiten
    size_images = [image.size]  # Originalgröße speichern

    # Bild in ein NumPy-Array umwandeln und vertikal stapeln
    data_array = np.vstack(np.asarray(image)) / constants.MAX_COLOR_VALUE  # Normalisieren
    st.write(f"Bildarray Form: {data_array.shape}")
    data_list = [data_array]

    # Fancy PCA Transformation
    fancy_pca_transform = FP.FancyPCA()

    st.subheader("Augmentierte Bilder:")
    cols = st.columns(constants.AUG_COUNT)  # Spalten für die Anzeige

    transformed_images = []  # Liste für Visualisierung sammeln

    for aug_count in range(constants.AUG_COUNT):
        try:
            # PCA auf das flache Bild anwenden
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array.copy())

            # Umwandlung auf den Bereich [0, 255] und zu uint8
            fancy_pca_images = (fancy_pca_images * 255).astype(np.uint8)

            # Rückumwandlung in die ursprüngliche Bildform: Höhe x Breite x 3
            height, width, channels = image.size[1], image.size[0], 3
            fancy_pca_images = fancy_pca_images.reshape((height, width, channels))

            st.write(f"Form des transformierten Bildes: {fancy_pca_images.shape}")

            # Speichern für die Visualisierung
            transformed_images.append(fancy_pca_images)
        except Exception as e:
            st.write(f"Fehler bei Fancy PCA: {e}")

        # Umwandlung zurück in Bildformat für die Anzeige
        try:
            aug_image = Image.fromarray(fancy_pca_images)

            # Anzeige der augmentierten Bilder in Spalten
            with cols[aug_count]:
                st.image(aug_image, caption=f"Augmentation {aug_count + 1}", use_container_width=True)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}")

    # t-SNE Visualisierung
    if transformed_images:
        st.subheader("t-SNE Visualisierung der augmentierten Bilder")

        # Bildverkleinerung für t-SNE
        MAX_PIXELS = 10000  # Maximale Anzahl Pixel für t-SNE
        SAMPLE_SIZE_FACTOR = 0.1  # 10% der Pixel für t-SNE-Berechnung

        def resize_image(image, max_pixels=MAX_PIXELS):
            """ Verkleinert das Bild, falls es zu groß ist. """
            width, height = image.size
            total_pixels = width * height
            if total_pixels > max_pixels:
                scale_factor = (max_pixels / total_pixels) ** 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            return image

        # Originalbild verkleinern
        image_resized = resize_image(image)

        # Transformierte Bilder für t-SNE vorbereiten
        
        image_original_array = np.array(image_resized).reshape(-1, 3)
        # Image.fromarray(img): mache das augmentierte Bild img zu einem PIL-Bild
        transformed_arrays = [np.array(resize_image(Image.fromarray(img))).reshape(-1, 3) for img in transformed_images] 

        # Pixel-Sampling für t-SNE
        sample_size_original = int(SAMPLE_SIZE_FACTOR * len(image_original_array))
        # len(image_original_array): Anzahl der Gesamtpixel; sample_size_original: Anzahl der gesamplten Pixel; replace=False: keine Duplikate bei der Pixelwahl
        image_original_sample = image_original_array[np.random.choice(len(image_original_array), sample_size_original, replace=False)] 

        # Macht genau das, was es für das Originalbild (Input-Bild) zuvor auch gemacht hat für die 5 augmentierten Bilder
        transformed_samples = []
        for img_array in transformed_arrays:
            sample_size_trans = int(SAMPLE_SIZE_FACTOR * len(img_array))
            transformed_samples.append(img_array[np.random.choice(len(img_array), sample_size_trans, replace=False)])

        # t-SNE Berechnung
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        original_tsne = tsne.fit_transform(image_original_sample) # t-SNE für Originalbild (Input-Bild)
        transformed_tsne = [tsne.fit_transform(sample) for sample in transformed_samples] # t-SNE für 5 augmentierte Bilder

        # Visualisierung mit Matplotlib
        fig, axs = plt.subplots(2, len(transformed_tsne) + 1, figsize=(12, 8))

        # Originalbild & t-SNE
        axs[0, 0].scatter(original_tsne[:, 0], original_tsne[:, 1], c=image_original_sample / 255, s=1)
        axs[0, 0].set_title("t-SNE Originalbild")
        axs[0, 0].axis("off")
        axs[1, 0].imshow(image_resized)
        axs[1, 0].axis("off")

        # Augmentierte Bilder & t-SNE
        for idx, tsne_result in enumerate(transformed_tsne):
            # tsne_result[:, 0]: x-Wert (erster Wert) jedes Punktes, tsne_result[:, 1]: y-Wert (zweiter Wert) jedes Punktes
            axs[0, idx + 1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=transformed_samples[idx] / 255, s=1)
            axs[0, idx + 1].set_title(f"t-SNE Augmentation {idx + 1}")
            axs[0, idx + 1].axis("off")
            axs[1, idx + 1].imshow(Image.fromarray(transformed_images[idx]))
            axs[1, idx + 1].axis("off")

        plt.tight_layout()
        st.pyplot(fig)