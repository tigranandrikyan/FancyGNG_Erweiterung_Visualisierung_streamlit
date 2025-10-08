import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import constants
from scipy.ndimage import gaussian_filter

#Interne Prozesse anzeigen lassen

### TODO: ggf. meinen Code zur Visualisierung (auch) hier einfügen/anpassen ### 

# 'view_nodes' zeigt eine Visualisierung von Knoten in verschiedenen Clustern
def view_nodes(finalNodes, node_cluster_map, cluster_count):
    color_matrix = [[] for _ in range(cluster_count)] # Erstellt leere 2D-Matrix (color_matrix) mit Anzahl an 'cluster_count' leeren Listen -> enthält Sammlung von Knoten für ein Cluster
    
    # Iteriert durch node_cluster_map und ordnet jedem Knoten (finalNodes[idx]) der richtigen Cluster-Liste in color_matrix zu, basierend auf der Cluster-ID (cluster_id)
    for idx, cluster_id in enumerate(node_cluster_map): # idx: Index; cluster_id: cluster-Wert zum zugehörigen Index
        color_matrix[int(cluster_id)].append(finalNodes[idx]) # z.B.: finalNodes=[[10,20], [30,40]], node_cluster_map=[0,1], cluster_count=2 -> color_matrix=[
        # [[10,20]], -> Cluster 0: Knoten [10,20]
        # [[30,40]]  -> Cluster 1: Knoten [30,40]
        # ]
    
    max_length = max(len(row) for row in color_matrix) # Berechnet die maximale Länge einer Liste innerhalb der color_matrix -> sicherstellen, dass alle Zeilen dieselbe Länge haben

    # Für jede Zeile in color_matrix wird die Zeile mit weißen Pixeln (RGB-Wert [255, 255, 255]) aufgefüllt -> sicherstellen, dass alle Zeilen gleiche Länge wie längste Zeile haben
    for row in color_matrix:
        row.extend([[255, 255, 255]] * (max_length - len(row)))
    
    color_matrix = np.array(color_matrix) # Konvertiert die color_matrix in ein NumPy-Array -> Daten für Bilddarstellung verwendbar
    
    plt.figure(figsize=(max_length, cluster_count)) # Erstellt eine neue Matplotlib-Figur mit einer Größe, die auf der maximalen Länge der Zeilen und der Anzahl der Cluster basiert
    plt.imshow(color_matrix, aspect='auto', interpolation="nearest") # aspect=auto: automatische Anpassung von Seitenverhältnis; interpolation=nearest: Pixel scharf (keine Interpolation)
    plt.axis('off')
    plt.show()

### TODO: ggf. meinen Code zur Augmentierung (auch) hier einfügen/anpassen ###

# Zeigt ein augmentiertes Bild an
def show_aug_image(aug_data, size_images):
    aug_image = aug_data * constants.MAX_COLOR_VALUE # Multipliziert die augmentierten Daten (aug_data) mit einem konstanten Wert MAX_COLOR_VALUE -> Farbbereich skalieren

    # Reshape zu RGB-Bild mit richtigen Dimensionen (Höhe, Breite, 3 für RGB); uint8: für Bilddarstellung erforderlich
    aug_image = aug_image.reshape((size_images[constants.DATA_INDEX][1], size_images[constants.DATA_INDEX][0], 3)).astype(np.uint8)
    image = Image.fromarray(aug_image)
    image.show()

# Zeigt ein Bild an, das den Glättungsprozess eines Clusters visualisiert
def show_smoothing(finalNodes, pixel_cluster_map, node_cluster_map, size_images, data_index):
    cluster_rep_color = _generate_cluster_rep_color(finalNodes, node_cluster_map) # berechnet repräsentative Farbe für jedes Cluster

    # Erstellt eine Liste (smoothing_field), die für jedes Pixel die repräsentative Cluster-Farbe aus cluster_rep_color auswählt
    smoothing_field = list()
    for cluster_idx in pixel_cluster_map:
        smoothing_field.append(cluster_rep_color[int(cluster_idx)])

    # Konvertiert smoothing_field in ein NumPy-Array und formt es in die Form eines RGB-Bildes um
    smoothing_field = np.array(smoothing_field)
    smoothing_field = smoothing_field.reshape((size_images[data_index][1], size_images[data_index][0], 3)) # shape: Höhe, Breite, RGB-Farben


    #before = _normalize_for_visualization(smoothing_field)

    # Zeigt das Bild vor der Glättung an
    before = Image.fromarray(smoothing_field.astype(np.uint8))
    before.show()

    smoothed_field = np.zeros_like(smoothing_field) # Erstellt ein leeres Array (smoothed_field) mit der gleichen Form wie smoothing_field zur Speicherung des infolge geglätteten Bildes

    # Wendet einen Gauß-Filter auf jede der drei Farbkanäle (RGB) an, um das Bild zu glätten
    for i in range(3): 
       smoothed_field[:, :, i] = gaussian_filter(smoothing_field[:, :, i], sigma=constants.SIGMA)
    
    #after = _normalize_for_visualization(smoothed_field)

    # Zeigt das Bild nach der Glättung an
    after = Image.fromarray(smoothed_field.astype(np.uint8))
    after.show()

# Diese Hilfsfunktion berechnet die durchschnittliche Farbe für jedes Cluster
def _generate_cluster_rep_color(finalNodes, node_cluster_map): # finalNodes: für die Berechnung der Farben; node_cluster_map: Zuordnung der Knoten zu Cluster-IDs
    sort_index_node_cluster_map = node_cluster_map.argsort() # Sortiert die node_cluster_map-IDs, um die Cluster in der Reihenfolge ihrer Knoten zu durchlaufen

    # Initialisiert Variablen für die repräsentative Farbe der Cluster und bereitet eine Liste der besuchten Cluster vor
    cluster_rep_color = list()
    taken_cluster = list()
    sum_cluster_nodes = np.array([0, 0, 0])
    count_cluster_nodes = 0
    taken_cluster.append(node_cluster_map[0])

    # Durchläuft alle Knoten und berechnet für jedes Cluster die durchschnittliche Farbe, indem die Farbwiederholungen summiert werden
    for node_index in sort_index_node_cluster_map: # Durchläuft alle Knoten in der sortierten Reihenfolge ihrer Clusterzuordnung
        if not (node_cluster_map[node_index] in taken_cluster): # Prüft, ob der aktuelle Cluster bereits verarbeitet wurde; wenn nicht, dann...
            taken_cluster.append(node_cluster_map[node_index]) # Fügt den neuen Cluster zur Liste der verarbeiteten Cluster hinzu
            cluster_rep_color.append((sum_cluster_nodes / count_cluster_nodes).astype(int)) # Berechnet die durchschnittliche Farbe für den vorherigen Cluster und speichert sie

            # Setzt die Summen und Zähler für den neuen Cluster zurück
            sum_cluster_nodes = np.array([0, 0, 0]) # Setzt den Farbwert-Summenvektor auf [0, 0, 0] zurück
            count_cluster_nodes = 0 # Setzt die Anzahl der Knoten im Cluster auf 0 zurück

        # Für jeden Knoten des Clusters wird die Farbe zur sum_cluster_nodes hinzugefügt und der Zähler count_cluster_nodes erhöht
        count_cluster_nodes += 1
        sum_cluster_nodes += finalNodes[node_index]
    
    # Berechnet und fügt die durchschnittliche Farbe für das letzte Cluster hinzu (in for-Schleife wird nur die durchschschn. Farbe des vorherigen, also bis zum vorletzten, berechnet)
    cluster_rep_color.append((sum_cluster_nodes / count_cluster_nodes).astype(int))

    
    return cluster_rep_color # Gibt die berechneten Farben für jedes Cluster zurück



