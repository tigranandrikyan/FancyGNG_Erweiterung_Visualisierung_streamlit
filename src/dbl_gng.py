#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import constants

# TODO: ändere etwas an den Hyperparametern in der __init__, um die Performance zu verbessern

class DBL_GNG():
    # Initialisierung der Klasse mit verschiedenen Hyperparametern
    def __init__(self, feature_number, maxNodeLength, L1=0.5, L2=0.01, 
                 errorNodeFactor = 0.5, newNodeFactor = 0.5):
       
        
        self.feature_number = feature_number # Anzahl der Merkmale (Features)
        self.M = maxNodeLength # Maximale Anzahl von Knoten
        self.alpha = L1 # Lernrate für den ersten Anpassungsschritt
        self.beta = L2 # Lernrate für den zweiten Anpassungsschritt          
        self.delta = errorNodeFactor # Fehlerverringerungsfaktor
        self.rho = newNodeFactor # Faktor für neue Knoten
        self.finalDistMap = 0 # Fürs Speichern der Zuordnung von jedem Datum zum nächstgelegenen Knoten (initialisiert mit dem Wert 0)
        
        self.eps = 1e-04 # Kleiner Wert zur Vermeidung von Division durch Null
        
      
        
    # Setzt Batch-Lernvariablen auf Null zurück, um keine veralteten Werte aus vorherigen Batches zu verwenden -> korrektes Lernen + stabile Netwerkaktualisierung
    def resetBatch(self):    
        self.Delta_W_1 = np.zeros_like(self.W) # Änderung der Gewichte für Schritt 1
        self.Delta_W_2 = np.zeros_like(self.W) # Änderung der Gewichte für Schritt 2
        self.A_1 = np.zeros(len(self.W)) # Aktivierungszähler für Schritt 1 (speichert, wie oft jeder Knoten in Schritt 1 aktiviert wurde)
        self.A_2 = np.zeros(len(self.W)) # Aktivierungszähler für Schritt 2
        self.S = np.zeros((len(self.W),len(self.W))) # Verbindungsstärke-Matrix             

    # Initialisiert das Netzwerk mit zufällig ausgewählten Startknoten    
    def initializeDistributedNode(self, data, number_of_starting_points = 1):
        
        data = data[:,:self.feature_number].copy() # Reduziert die Daten auf relevante Merkmale
        np.random.shuffle(data) # Durchmischt die Daten zufällig
        
        nodeList = np.empty((0,self.feature_number),dtype=np.float32) # Liste der Knoten
        edgeList = np.empty((0,2),dtype=int) # Liste der Kanten

        
        #copy the data, ready for crop
        tempData = data.copy()

        #define the batch size (Größe der Teilmengen)
        batchSize = len(data) // number_of_starting_points

      
       
        for i in range(number_of_starting_points):
            idx = np.arange(len(tempData), dtype=int) # Indexliste für Daten
            
            #randomly select a node
            selectedIndex = np.random.choice(idx[-batchSize:]) # Zufällige Auswahl eines Index
            currentNode = tempData[selectedIndex] # Auswahl eines Startknotens
            
            # insert the node into list
            nodeList = np.append(nodeList, [currentNode],axis=0)
            
            # calculate the distance from all data to the current node (Berechnet Distanzen zu allen anderen Punkten)
            y2 = np.sum(np.square(tempData), axis=1)
            dot_product = 2 * np.matmul(currentNode,tempData.T)
            dist =  y2 - dot_product 
            idx = np.argsort(dist) # # perform sorting, so now we know which is closest, which is farthest (Sortiert nach Distanz)
            
            
            # select the third cloest node as neighbor, try to leave some space in between
            neighborNode = tempData[idx[2]] # Wählt den dritt-nächsten Punkt als Nachbarn, um etwas Abstand zwischen den Knoten zu bewahren
            # add neighbor node into the list
            nodeList = np.append(nodeList, [neighborNode],axis=0) # Fügt den Nachbarn zur Knotenliste hinzu
            
            # connect them and add to the list (Fügt eine Kante zwischen dem aktuellen Knoten und dem Nachbarn hinzu)
            edgeList = np.append(edgeList,[[i*2, i*2 + 1]],axis=0) # Bsp.: für i=0 -> [0, 1], für i=1 -> [2, 3], usw., d.h. Kanten zwischen den Knotenpaaren
            
            #randomly select a node from the farthest nodes within the batch size
            selectedIndex = np.random.choice(idx[-batchSize:])            
            currentNode = tempData[selectedIndex,:2]
            
            # remove the current area, so it won't repeat in the follow search (Entfernt die bereits verarbeiteten Punkte aus den Daten, damit sie nicht wieder verwendet werden)
            idx = idx[batchSize:]
            tempData = tempData[idx]
        
       
        self.W = nodeList # Speichert die Knoten
        self.C = edgeList # Speichert die Kanten
        self.E = np.zeros(len(self.W)) # Initialisiert Fehlerwerte für Knoten
    
         
       
    #input Batch, Feature (Führt einen Lernzyklus mit einem Batch von Eingabedaten durch)
    def batchLearning(self, X):
        X = X[:,:self.feature_number] # Verwendet nur die relevanten Merkmale -> alle Zeilen, nur die ersten `feature_number` Spalten
        
        # identity Matrix
        i_adj = np.eye(len(self.W)) # Identitätsmatrix für Nachbarschaftsbeziehungen
        
        adj = np.zeros((len(self.W),len(self.W))) # Adjazenzmatrix (Verbindungen)

        # Kanten in beide Richtungen setzen, damit Richtung egal ist (ungerichteter Graph)
        adj[self.C[:,0],self.C[:,1]] += 1 # Kanten setzen (in eine Richtung)
        adj[self.C[:,1],self.C[:,0]] += 1 # Kanten setzen (in die andere Richtung)

        adj[adj > 0] = 1 # Binär machen (nur 0 oder 1), um eine Adjazenzmatrix zu erhalten       
        
        
        batchIndices = np.arange(len(X))
        
        
        #obtain Distance
        x2 = np.sum(np.square(X), axis=1) # Berechnet quadrierte Werte für Eingaben
        y2 = np.sum(np.square(self.W), axis=1) # Quadrierte Werte für Knoten             
        dot_product = 2 * np.matmul(X, self.W.T) # Skalarprodukt   
      
        dist = np.clip(np.expand_dims(x2, axis=1) + y2 - dot_product, a_min=0, a_max = None) # Distanzberechnung: sqrt(x^2+y^2 -2<x,y>)
        dist = np.sqrt(dist  + self.eps) # Verhindert Null-Division
        #Dist col = node, row = datum
        
                
        # get fist and second winner nodes, s1: row-wise min distance node index, for each datum.
        tempDist = dist.copy()                
        s1 = np.argmin(tempDist,axis=1) # Nächster Knoten
        tempDist[batchIndices,s1] = 99999 # Setzt die Distanz zum nächsten Knoten (Matrixeintrag in tempDist) auf einen hohen Wert, um als Nächstes den zweitnächsten Knoten zu finden              
        s2 = np.argmin(tempDist,axis=1) # Zweitnächster Knoten, da Distanz zum nächstgelegenen Knoten auf 99999 gesetzt wurde
        
        # add error to s1
        self.E += np.sum(i_adj[s1] * dist, axis=0) * self.alpha # Fehlerwerte aktualisieren
        
        # Update s1 position
        self.Delta_W_1 += (np.matmul(i_adj[s1].T, X) - (self.W.T * np.sum(i_adj[s1],axis=0)).T)  * self.alpha
        
        
        # Update s1 neighbor position        
        self.Delta_W_2 += (np.matmul(adj[s1].T, X) - np.multiply(self.W.T, adj[s1].sum(0)).T) * self.beta
    
        
        # Add 1 to s1 node activation
        self.A_1 += np.sum(i_adj[s1], axis=0) # Aktivierung aktualisieren
                
        # Add 1 to neighbor node acitvation
        self.A_2 += np.sum(adj[s1], axis=0) # Aktivierung aktualisieren
        

        # Count the important edge (s1 and s2)
        connectedEdge = np.zeros_like(self.S)              
        connectedEdge[s1,s2] = 1 # Verbindet Knoten s1 und s2
        connectedEdge[s2,s1] = 1 # Verbindet Knoten s2 und s1
        # Verbindungen in beide Richtungen, um ungerichteten Graphen zu erhalten

        t = i_adj[s1] + i_adj[s2]
        connectedEdge *= np.matmul(t.T,t) # Aktualisiert Verbindungsstärken
        
        self.S += connectedEdge # Speichert neue Verbindungen
        
    
    # Aktualisiert das Netzwerk nach einem Lernzyklus    
    def updateNetwork(self):
        
       
        self.W += (self.Delta_W_1.T * (1 / (self.A_1 + self.eps))).T + (self.Delta_W_2.T * (1 / (self.A_2 + self.eps))).T # Aktualisiert Knotenpositionen       
        
        self.C = np.asarray(self.S.nonzero()).T # Setzt die neue Kantenliste
      
        
        
        self.removeIsolatedNodes() # Entfernt isolierte Knoten
        
        
        self.E *= self.delta # Fehlerwerte verringern
        
        if random.random() > 0.9: # Zufällige Bedingung für eine Bereinigung zur kontrollierten Entfernung von Knoten
            self.removeNonActivatedNodes() # Entfernt nicht aktivierte Knoten
        
        
    def removeIsolatedNodes(self):
        # Erstellt eine Adjazenzmatrix für das Netzwerk
        adj = np.zeros((len(self.W),len(self.W)))

        # Setzt Einträge in der Adjazenzmatrix für bestehende Verbindungen auf 1 (ungerichteter Graph)
        adj[self.C[:,0],self.C[:,1]] = 1
        adj[self.C[:,1],self.C[:,0]] = 1
        
        # Findet isolierte Knoten, die keine Verbindungen haben
        isolatedNodes = (np.sum(adj, axis=0) + np.sum(adj, axis=1) == 0).nonzero()[0] # [0] = erste Array des Tupels (Zeile, Spalte), d.h. Zeilenindices
        
        # Speichert die zu löschenden Knoten in einer sortierten Liste
        finalDelete = list(np.unique(isolatedNodes))
        if len(finalDelete) > 1:            
            finalDelete.sort(reverse=True) # Sortiert rückwärts, um Indexverschiebungen zu vermeiden  
        
        # Entfernt die isolierten Knoten aus der Kantenliste 'C'
        for v in finalDelete:                       
            self.C = self.C[np.logical_not(np.logical_or(self.C[:,0] == v, self.C[:,1] == v))] # Bedingte Filterung

            # Reduziert die Knotenindizes in der Kantenliste           
            self.C[self.C[:,0] > v,0] -= 1
            self.C[self.C[:,1] > v,1] -= 1       
            
        # Löscht die isolierten Knoten aus allen relevanten Datenstrukturen
        if len(finalDelete) > 0:
            #print("Isolated",finalDelete) -> Debugging
            self.S = np.delete(self.S, finalDelete,axis=0)
            self.S = np.delete(self.S, finalDelete,axis=1)

            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)
        
        
    def removeNonActivatedNodes(self):
        
        # Ermittelt die Knoten, die nicht aktiviert wurden
        nodeActivation = self.A_1 
        nonActivatedNodes = (nodeActivation == 0).nonzero()[0]
        
        # Speichert die zu löschenden Knoten
        finalDelete = list(nonActivatedNodes)
        if len(finalDelete) > 1:                
            finalDelete.sort(reverse=True) # Sortiert rückwärts, um Indexverschiebungen zu vermeiden  
        
        # Entfernt inaktive Knoten aus der Kantenliste 'C'
        for v in finalDelete:    
           
            self.C = self.C[np.logical_not(np.logical_or(self.C[:,0] == v, self.C[:,1] == v))]
            
            self.C[self.C[:,0] > v,0] -= 1
            self.C[self.C[:,1] > v,1] -= 1           

        # Löscht die inaktiven Knoten aus allen relevanten Datenstrukturen    
        if len(finalDelete) > 0:
            #print("Non Activated",finalDelete) -> Debugging
            
            
            self.S = np.delete(self.S, finalDelete,axis=0)
            self.S = np.delete(self.S, finalDelete,axis=1)
            
            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)
  
    
    def addNewNode(self, gng):

        # Berechnet, wie viele neue Knoten basierend auf der Fehlerenergie 'E' hinzugefügt werden sollen

        g = np.sum(self.E > np.quantile(gng.E,0.85)) # Anzahl der Knoten in self.E (True=1 in der Abfrage) mit Fehler über dem 85. Perzentil (15% der Knoten) von gng.E
        
        for _ in range(g):

            # Falls die maximale Knotenzahl erreicht ist, abbrechen
            if len(self.W) >= self.M:
                return # 'return' beendet die Methode, d.h. der Code soll an dieser Stelle abbrechen -> Methode addNewNode soll nicht weiter ausgeführt werden
            
            # Wählt den Knoten mit dem höchsten Fehlerwert aus
            q1 = np.argmax(self.E)
            if self.E[q1] <= 0:          
                #print("Zero error q1") -> Debugging
                return
            
            # get the connected nodes (Findet alle direkt mit `q1` verbundenen Knoten)
            connectedNodes = np.unique(np.concatenate((self.C[self.C[:,0] == q1,1], self.C[self.C[:,1] == q1,0]))) # sucht nach allen eindeutigen Nachbarn von 'q1' 
            if len(connectedNodes) == 0:
                return
           
            
            # get the maximum error of neighbors (Wählt den verbundenen Knoten mit dem höchsten Fehlerwert aus)
            q2 = connectedNodes[np.argmax(self.E[connectedNodes])]
            if self.E[q2] <= 0:              
                #print("Zero error q2") -> Debugging
                return
          
            
            # insert new node between q1 and q2 (Erzeugt einen neuen Knoten als Mittelwert zwischen 'q1' und 'q2')
            q3 = len(self.W) # Indexposition des neuen Knotens 'new_w' ('q1' und 'q2' sind ebenfalls Indexpositionen der verbundenen Knoten)
            new_w = (self.W[q1] + self.W[q2]) * 0.5 # Mittelwertberechnung        
            self.W = np.vstack((self.W, new_w)) # Fügt den neuen Knoten zu 'W' als neue Zeile hinzu

            # Fügt den Fehlerwert=0 des neuen Knotens 'new_w' in die Fehlerliste 'E' ein (0, da er gerade durch die Mittelwertberechnung erstellt wurde und als ideal angesehen wird)           
            self.E = np.concatenate((self.E,np.zeros(1)),axis=0)
            
           
            # update the error (Reduziert die Fehlerwerte von 'q1' und 'q2' (gemeint sind im gesamten Code die Werte an den Positionen von 'q1' und 'q2', wenn von "Werte von q1, q2, q3" die Rede ist))
            self.E[q1] *= self.rho
            self.E[q2] *= self.rho        
            self.E[q3] = (self.E[q1] + self.E[q2]) * 0.5 # Neuer Fehlerwert für 'q3'
            
         
                       
            #remove the original edge (Entfernt die Verbindung zwischen 'q1' und 'q2', unabhängig von der Reihenfolge der Knoten (ungerichteter Graph) - wie in DBL_GNG beschrieben)
            self.C = self.C[np.logical_not(np.logical_and(self.C[:,0] == q1, self.C[:,1] == q2))]
            self.C = self.C[np.logical_not(np.logical_and(self.C[:,0] == q2, self.C[:,1] == q1))]
            
        
            #add the edge (Erstellt neue Verbindungen für 'q3')
            self.C = np.vstack((self.C,np.asarray([q1,q3])))
            self.C = np.vstack((self.C,np.asarray([q2,q3])))
       
            # Fügt 'q3' zur Adjazenzmatrix 'S' hinzu     
            self.S = np.pad(self.S, pad_width=((0, 1), (0, 1)), mode='constant') # add a col and row (Auffüllen der Matrix mit einer zusätzlichen Zeile und Spalte am Ende jeweils)
            self.S[q1,q2] = 0
            self.S[q2,q1] = 0
            self.S[q1,q3] = 1
            self.S[q3,q1] = 1
            self.S[q2,q3] = 1
            self.S[q3,q2] = 1  
            
            # Aktualisiert die Aktivierungswerte
            self.A_1 = np.concatenate((self.A_1,np.ones(1)),axis=0)  
            self.A_2 = np.concatenate((self.A_2,np.ones(1)),axis=0)  
        
       
    def cutEdge(self):
        # Entfernt nicht aktivierte Knoten  
        self.removeNonActivatedNodes()
        
        #self.S = self.S.astype(int) -> Debugging

        # Erstellt eine Maske (boolesche Matrix, in der jede Zelle True ist, wenn dort eine Kante existiert, sonst False) für bestehende Kanten
        mask = self.S > 0

        #print(self.S[mask]) -> Debugging

        # Bestimmt den Schwellenwert für das Abschneiden von Kanten
        filterV = np.quantile(self.S[mask], constants.EDGE_CUTTING) # (constants.EDGE_CUTTING)-Perzentil der Gewichte

        #print(filterV) -> Debugging
        #print(self.S) -> Debugging

        # Erstellt eine Kopie der Adjazenzmatrix
        temp = self.S.copy()

        #print(temp) -> Debugging

        # Setzt schwache Verbindungen auf 0
        temp[self.S < filterV] = 0

        #print(temp) -> Debugging
        #print(self.C) -> Debugging

        # Extrahiert die verbleibenden Kanten
        self.C = np.asarray(temp.nonzero()).T

        #print(self.C) -> Debugging
        
        # Entfernt nun isolierte Knoten, die durch das Abschneiden entstanden sind
        self.removeIsolatedNodes()
    
    
    # Berechnet für jedes Eingabedatum den nächstgelegenen Knoten basierend auf der euklidischen Distanz
    def finalNodeDatumMap(self, X):
        X = X[:,:self.feature_number] # Beschränkt X auf die ersten 'feature_number' Spalten (nur relevante Merkmale beibehalten)        
 
        #obtain Distance
        x2 = np.sum(np.square(X), axis=1) # Berechnung der quadratischen Summen der Eingabedaten 'X' für jede Zeile (||X||^2)
        y2 = np.sum(np.square(self.W), axis=1) # Berechnung der quadratischen Summen der Gewichte 'W' für jede Zeile (||W||^2)          
        dot_product = 2 * np.matmul(X, self.W.T) # Berechnung des doppelten Skalarprodukts (Matrixmultiplikation = Sammlung von SPen) zwischen X und den transponierten Gewichten 'W' (2 * <X,W^T>)  

        # Berechnung der euklidischen Distanzmatrix zwischen den Eingabedaten und den Gewichten:
        # dist = ||X||² + ||W||² - 2 * (X @ Wᵀ), dabei wird sichergestellt, dass der minimale Wert 0 ist (keine negativen Werte) durch Clippen
        dist = np.clip(np.expand_dims(x2, axis=1) + y2 - dot_product, a_min=0, a_max = None)

        dist = np.sqrt(dist  + self.eps) # Wurzel ziehen zur Berechnung der tatsächlichen euklidischen Distanz (unter Berücksichtigung von 'eps', um Division durch Null zu vermeiden)
        # HINWEIS: 'dist' ist jetzt eine Distanzmatrix, in der jede Zeile einem Eingabedatum und jede Spalte einem Knoten entspricht

        #Dist col = node, row = datum
        
                
        # get fist and second winner nodes, s1: row-wise min distance node index, for each datum.
        tempDist = dist.copy() # Erstellen einer Kopie der Distanzmatrix, um das Original nicht zu verändern               
        s1 = np.argmin(tempDist,axis=1) # Finden des Index des nächstgelegenen Knotens (geringste Distanz) für jede Zeile (jedes Datum)
        self.finalDistMap = s1 # Speichern der Zuordnung von jedem Datum zum nächstgelegenen Knoten 
        
  










