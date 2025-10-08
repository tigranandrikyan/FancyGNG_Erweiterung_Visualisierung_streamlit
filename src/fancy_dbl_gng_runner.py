import dbl_gng 
import constants
import clustering
import view
import parser
import color_pca
import numpy as np
from tqdm import trange

#Starten von fancy dbl gng

file_list = parser.generate_file_list() # Erstellen einer Liste von Dateien
#print(f"Dateiliste: {file_list}")  ### Debugging: Überprüfe den Inhalt der Dateiliste ###
data, size_images = parser.parse(file_list) # Parsen der Dateien und Extrahieren der Bilddaten und Bildgrößen
#print(size_images[0], data[0].shape)
data_names = parser.get_image_names(file_list) # Abrufen der Bildnamen aus der Dateiliste
#print(data_names) # Ausgabe der Bildnamen zur Überprüfung

# Initialisierung von Listen zur Speicherung der Cluster-Anzahlen und der endgültigen Knotenzahlen
cluster_counts = list()
final_nodes_count = list()

epoch = constants.EPOCH # Konstante 'EPOCH' aus constants.py

# Initialisierung von Variablen zur Speicherung der finalen Distanz und der endgültigen Knoten
finalDist = 0
finalNodes = 0

# Durchlaufen aller Dateien in der Liste
for data_index in range(len(file_list)):
    #if data_names[data_index] != 'blumenwiese2':
    #    continue
    #print(data[data_index].max(), data[data_index].min())
    gng = dbl_gng.DBL_GNG(3,constants.MAX_NODES) # Erstellen eines DBL_GNG-Objekts mit 3 Dimensionen und einer Maximalanzahl Knoten, gegeben durch 'MAX_NODES' aus constants.py
    gng.initializeDistributedNode(data[data_index], constants.SARTING_NODES) # Initialisierung der verteilten Knoten mit den Daten der aktuellen Datei, Fehler: STARTING_NODES

    # Durchführen der Trainings-Epochen
    bar = trange(epoch) # Fortschrittsbalken für die Epoche
    for i in bar:
        #print(f"Epoch: {i + 1}") # Ausgabe der aktuellen Epoche -> Debugging
        gng.resetBatch() # Zurücksetzen des Batch-Trainingsstatus
        gng.batchLearning(data[data_index]) # Durchführen des Batch-Lernens
        gng.updateNetwork() # Aktualisieren des Netzwerks
        gng.addNewNode(gng) # Hinzufügen neuer Knoten
        bar.set_description(f"Epoch: {data_names[data_index]}, Knotenanzahl: {len(gng.W)}") # Anzeige der aktuellen Epoche im Fortschrittsbalken
    gng.cutEdge() # Entfernen nicht benötigter Verbindungen im Netzwerk

    gng.finalNodeDatumMap(data[data_index]) # Erstellen einer finalen Zuordnung der Knoten zu den Datenpunkten
    finalDistMap = gng.finalDistMap # Abrufen der finalen Distanz-Matrix
    finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int) # Berechnung der endgültigen Knotenwerte und Skalierung mit dem maximalen Farbwert 'MAX_COLOR_VALUE' aus constants.py
    connectiveMatrix = gng.C # Abrufen der Verbindungs-Matrix zwischen Knoten
    pixel_cluster_map, node_cluster_map  = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix) # Durchführung des Clusterings basierend auf den finalen Knoten und Distanzen
    pixel_cluster_map = np.array(pixel_cluster_map) # Umwandlung der Cluster-Zuordnung in ein numpy-Array
    #print(np.unique(np.array(pixel_cluster_map)))
    #print(data[0].shape)
    #print(np.array(pixel_cluster_map))
    cluster_count = int(max(node_cluster_map)) + 1 # Berechnung der Anzahl der Cluster (maximale Knoten-ID + 1)
    #print(cluster_count)

    # Speichern der Anzahl der Cluster und der Anzahl der finalen Knoten
    cluster_counts.append(int(cluster_count))
    final_nodes_count.append(len(finalNodes) + 1)

    # Debugging: Visualisierung des Netzwerks:
    #view.view_nodes(finalNodes, node_cluster_map, cluster_count)

    # Durchführung mehrerer Datenaugmentierungen -> TODO: ggf. meinen Code zur Augmentierung (auch) hier einfügen/anpassen
    for aug_count in trange (constants.AUG_COUNT):
        #print(f"Augmentation: {str(data_names[data_index])}, {aug_count + 1}/{constants.AUG_COUNT}") # Debugging
        aug_data = color_pca.modify_clusters(data[data_index], pixel_cluster_map, cluster_count, size_images, data_index) # Modifikation der Clusterfarben durch Hauptkomponentenanalyse (PCA)
        parser.save_data(aug_data, data_names[data_index], size_images, aug_count, data_index, path = './out_data') # Speichern der augmentierten Daten im Ausgabeordner

        ### Debugging: Visualisierung der Glättung der Knoten und Cluster:
        #view.show_smoothing(finalNodes, pixel_cluster_map, node_cluster_map, size_images, data_index)    
    

