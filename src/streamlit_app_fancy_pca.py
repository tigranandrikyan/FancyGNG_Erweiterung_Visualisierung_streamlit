import streamlit as st
import numpy as np
import constants
import fancy_pca as FP
from PIL import Image

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
    st.write(f"Dateityp: {type(uploaded_file)}") ### Debugging: Zeigt den Dateityp an ###
    try:
        image = Image.open(uploaded_file) # Öffnet das Bild mit PIL

        # Prüfe, ob RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        st.write(f"Bildgröße: {image.size}") ### Debugging: Zeigt die Größe des Bildes an ###
    except Exception as e:
        st.write(f"Fehler beim Öffnen des Bildes: {e}") ### Debugging ###

    # Bild für Fancy PCA vorbereiten -> wird nicht benutzt (ist ein Überbleibsel und wurde ursprünglich angelehnt an parser.py erstellt)
    size_images = [image.size]  # Originalgröße speichern

    # Bild in ein NumPy-Array umwandeln und vertikal stapeln und normalisieren (wie in parser.py)
    data_array = np.vstack(np.asarray(image)) / constants.MAX_COLOR_VALUE  # besser: data_array = np.asarray(image).reshape(-1, 3) / constants.MAX_COLOR_VALUE
 
    st.write(f"Bildarray Form: {data_array.shape}") ### Debugging: Gibt die Form des Bildarrays aus ###
    data_list = [data_array]

    # Fancy PCA Transformation
    fancy_pca_transform = FP.FancyPCA()

    st.subheader("Augmentierte Bilder:")
    cols = st.columns(constants.AUG_COUNT)  # Spalten für die Anzeige der augmentierten Bilder

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

            st.write(f"Form des transformierten Bildes: {fancy_pca_images.shape}") # Debugging: Gibt die Form des transformierten Bildes aus
        except Exception as e:
            st.write(f"Fehler bei Fancy PCA: {e}") ### Debugging ### 

        # Konvertiert das NumPy-Array zurück in ein Bildformat zur Anzeige
        try:
            aug_image = Image.fromarray(fancy_pca_images)

            # Zeigt die augmentierten Bilder in den zuvor erstellten Spalten an
            with cols[aug_count]:
                st.image(aug_image, caption=f"Augmentation {aug_count + 1}", use_container_width=True)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}") ### Debugging ###