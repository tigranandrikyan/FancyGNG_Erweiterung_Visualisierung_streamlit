import constants
import parser
import fancy_pca as FP


#Starten von fancy pca

file_list = parser.generate_file_list() # Erzeugt eine Liste der Dateien, die verarbeitet werden sollen
data, size_images = parser.parse(file_list) # Parst die Dateien und extrahiert die Bilddaten sowie deren Größen
data_names = parser.get_image_names(file_list) # Ruft die Namen der Bilder aus der Dateiliste ab

### Debugging: Ausgabe der Bildnamen: ###
#print(data_names)

fancy_pca_transform = FP.FancyPCA() # Erstellt eine Instanz der FancyPCA-Transformation (Objekt der Klasse FancyPCA())

# Iteriert über alle Dateien in der Liste
for data_index in range(len(file_list)):

    # Wiederholt die Augmentierung für die festgelegte Anzahl an Augmentierungen durch 'AUG_COUNT' aus constants.py -> TODO: ggf. meinen Code zur Augmentierung (auch) hier einfügen/anpassen
    for aug_count in range (constants.AUG_COUNT):
        print(f"Fancy_PCA: {str(data_names[data_index])}, {aug_count + 1}/{constants.AUG_COUNT}") ### Debugging: Gibt eine Statusmeldung mit Bildname und Augmentierungsfortschritt aus ###
        fancy_pca_images = fancy_pca_transform.fancy_pca(data[data_index]) # Wendet die FancyPCA-Transformation auf das aktuelle Bild an

        # Speichert das transformierte Bild mit den gegebenen Parametern im Ausgabeordner
        parser.save_data(fancy_pca_images, data_names[data_index], size_images, aug_count, data_index, path = './out_fancy_pca')