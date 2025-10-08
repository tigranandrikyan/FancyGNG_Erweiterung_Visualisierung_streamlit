import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------- GNG Bildverarbeitung --------------------------

# Streamlit UI
st.title("DBL-GNG Image Augmentation")
st.write("Lade ein Bild hoch oder nimm eines mit der Kamera auf.")
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

if input_option == "Datei-Upload":
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
elif input_option == "Kamera":
    uploaded_file = st.camera_input("Bild aufnehmen")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}")

        image_array = np.asarray(image)
        data_array = image_array.reshape(-1, 3) / constants.MAX_COLOR_VALUE
        st.write(f"Bildarray Form: {data_array.shape}")

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

    except Exception as e:
        st.write(f"Fehler beim Öffnen oder Verarbeiten des Bildes: {e}")