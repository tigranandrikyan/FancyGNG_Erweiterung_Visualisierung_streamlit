import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Titel und Beschreibung
st.title("DBL-GNG Image Augmentation")
st.write("Lade ein Bild hoch oder nimm eines mit der Kamera auf.")

# Bildquelle wählen
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

# Upload oder Kameraaufnahme
uploaded_file = None
if input_option == "Datei-Upload":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
elif input_option == "Kamera":
    uploaded_file = st.camera_input("Bild aufnehmen")

if uploaded_file is not None:
    try:
        # Einheitliche Behandlung der Bilddaten als Bytes
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}")

        image_array = np.asarray(image)
        data_array = image_array.reshape(-1, 3) / constants.MAX_COLOR_VALUE

        st.write(f"Bildarray Form: {data_array.shape}")

        # DBL-GNG initialisieren
        gng = dbl_gng.DBL_GNG(3, constants.MAX_NODES)
        gng.initializeDistributedNode(data_array, constants.SARTING_NODES)

        bar = trange(constants.EPOCH)
        for i in bar:
            gng.resetBatch()
            gng.batchLearning(data_array)
            gng.updateNetwork()
            gng.addNewNode(gng)
            bar.set_description(f"Epoch {i + 1} Knotenanzahl: {len(gng.W)}")

        gng.cutEdge()
        gng.finalNodeDatumMap(data_array)

        finalDistMap = gng.finalDistMap
        finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int)
        connectiveMatrix = gng.C

        pixel_cluster_map, node_cluster_map = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix)
        pixel_cluster_map = np.array(pixel_cluster_map)
        cluster_count = int(max(node_cluster_map)) + 1

        st.write(f"Anzahl der Cluster: {cluster_count}")

        cols = st.columns(constants.AUG_COUNT + 1)
        all_images = [image]

        for aug_count in trange(constants.AUG_COUNT):
            try:
                aug_data = color_pca.modify_clusters(data_array, pixel_cluster_map, cluster_count, [image.size], 0)
                aug_data = (aug_data * 255).astype(np.uint8)
                aug_data = aug_data.reshape((image.size[1], image.size[0], 3))
                aug_image = Image.fromarray(aug_data)
                all_images.append(aug_image)
            except Exception as e:
                st.write(f"Fehler bei der Augmentierung: {e}")

        # t-SNE Visualisierung
        if all_images:
            st.subheader("t-SNE Visualisierung der Bilder")

            MAX_PIXELS = 10000
            SAMPLE_SIZE_FACTOR = 0.1

            def resize_image(image, max_pixels=MAX_PIXELS):
                width, height = image.size
                total_pixels = width * height
                if total_pixels > max_pixels:
                    scale_factor = (max_pixels / total_pixels) ** 0.5
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                return image

            image_resized = resize_image(image)
            image_original_array = np.array(image_resized).reshape(-1, 3)
            transformed_arrays = [np.array(resize_image(img)).reshape(-1, 3) for img in all_images]

            sample_size_original = int(SAMPLE_SIZE_FACTOR * len(image_original_array))
            image_original_sample = image_original_array[np.random.choice(len(image_original_array), sample_size_original, replace=False)]

            transformed_samples = []
            for img_array in transformed_arrays:
                sample_size_trans = int(SAMPLE_SIZE_FACTOR * len(img_array))
                transformed_samples.append(img_array[np.random.choice(len(img_array), sample_size_trans, replace=False)])

            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            original_tsne = tsne.fit_transform(image_original_sample)
            transformed_tsne = [tsne.fit_transform(sample) for sample in transformed_samples]

            fig, axs = plt.subplots(2, len(transformed_tsne), figsize=(14, 8))

            axs[0, 0].scatter(original_tsne[:, 0], original_tsne[:, 1], c=image_original_sample / 255, s=1)
            axs[0, 0].set_title("t-SNE Original")
            axs[0, 0].axis("off")
            axs[1, 0].imshow(image_resized)
            axs[1, 0].set_title("Originalbild")
            axs[1, 0].axis("off")

            for idx, tsne_result in enumerate(transformed_tsne[1:]):
                axs[0, idx + 1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=transformed_samples[idx + 1] / 255, s=1)
                axs[0, idx + 1].set_title(f"t-SNE Aug. {idx + 1}")
                axs[0, idx + 1].axis("off")
                axs[1, idx + 1].imshow(all_images[idx + 1])
                axs[1, idx + 1].set_title(f"Augmentiertes Bild {i}")
                axs[1, idx + 1].axis("off")

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.write(f"Fehler beim Öffnen oder Verarbeiten des Bildes: {e}")