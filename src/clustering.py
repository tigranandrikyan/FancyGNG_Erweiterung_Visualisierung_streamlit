import numpy as np
from collections import defaultdict # Importiert defaultdict aus collections für die Speicherung von Adjazenzlisten

#Hier wird den Pixeln die Gruppe zugeordnet


# Diese Funktion erstellt Cluster-Gruppen basierend auf den gegebenen Kanten
def _createrClusterGroups(edges): # '_' bedeutet, dass die Funktion nur innerhalb des Moduls verwendet wird
    graph = defaultdict(list) # Erstellt ein Wörterbuch, in dem jeder Knoten eine Liste von Nachbarn hat -> Beispiel: {1: [2], 2: [1]}

    # Erstellt eine ungerichtete Adjazenzliste aus den gegebenen Kanten
    for edge in edges: # Beispiel: edge = (1, 2)
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0]) 

    
    # Tiefensuche (Depth-First Search, dfs), um alle verbundenen Knoten einer Komponente zu finden
    def dfs(node, visited):
        component = [] # Liste zur Speicherung der aktuellen zusammenhängenden Komponente
        stack = [node] # Stack für die Tiefensuche: Füge den aktuellen Knoten hinzu
        visited.add(node) # Markiert den aktuellen Knoten als besucht

        while stack: # Solange noch Knoten im Stack sind (True), führe die Tiefensuche fort
            current = stack.pop() # Nimmt den obersten Knoten aus dem Stack, entferne aus der Liste und gib es zurück
            component.append(current) # Fügt den Knoten zur aktuellen Komponente hinzu

            for neighbor in graph[current]: # Durchläuft alle Nachbarn des aktuellen Knotens
                if neighbor not in visited: # Falls der Nachbar noch nicht besucht wurde
                    visited.add(neighbor) # Markiert den Nachbarn als besucht
                    stack.append(neighbor) # Fügt den Nachbarn dem Stack hinzu
                    
        return component # Gibt die Liste der zusammenhängenden Knoten zurück -> Graphentheorie: zusammenhängender Graph -> alle Knoten paarweise durch eine Kantenfolge verbunden

    # Findet alle Gruppen zusammenhängender Knoten im Graphen
    def find_all_connected_groups():
        visited = set() # set() zur Speicherung der bereits besuchten Knoten
        all_groups = [] # Liste zur Speicherung aller gefundenen Cluster

        # Durchläuft alle Knoten im Graphen
        for node in graph:
            if node not in visited: # Falls der Knoten noch nicht besucht wurde
                component = dfs(node, visited) # Finde alle verbundenen Knoten der aktuellen Komponente 'node'
                all_groups.append(np.array(component)) # Speichert die Komponente als NumPy-Array in der Liste

        return all_groups # Gibt die Liste aller Gruppen zurück

    # Gibt die gefundenen Cluster zurück: find_all_connected_groups() gibt all_groups zurück und _createClusterGroups() gibt find_all_connected_groups() zurück -> Rückgabe: all_groups
    return find_all_connected_groups() 

# Erstellt eine Zuordnung von Knoten zu Cluster-IDs
def _createClusterNodeMap(connected_nodes, finalNodes):
    node_cluste_map = np.zeros(len(finalNodes)) # Erstellt ein Array mit Nullen der Länge finalNodes -> Speichert die Cluster-IDs der Knoten

    # Iteriert über die gefundenen Cluster und weist jedem Knoten eine Cluster-ID zu
    for cluster_index, cluster in enumerate(connected_nodes):
          for node in cluster:
              node_cluste_map[node] = cluster_index # Speichert die Cluster-Nummer für den jeweiligen Knoten
    return node_cluste_map # Gibt die Cluster-Zuordnung zurück           
      

### Hauptfunktion zur Cluster-Bildung basierend auf der Datenstruktur ###
def cluster(datum_node_map, finalNodes, edges):
    connected_nodes = _createrClusterGroups(edges) # Findet zusammenhängende Gruppen von Knoten
    #print(connected_nodes) -> Debugging
    node_cluster_map = _createClusterNodeMap(connected_nodes, finalNodes) # Erstellt eine Cluster-Zuordnung für die Knoten
    #print(node_cluster_map) -> Debugging
    pixel_cluster_map = list() # Liste zur Speicherung der Cluster-Zuordnung für die Pixel

    # Weist jedem Datenpunkt aus datum_node_map ein Cluster zu
    for node_index in datum_node_map:
        pixel_cluster_map.append(node_cluster_map[node_index])

    return pixel_cluster_map, node_cluster_map # Gibt die Pixel- und Knoten-Cluster-Zuordnung zurück