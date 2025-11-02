import numpy as np
import constants

#Fancy PCA

### TODO: aktuell: Eigenwertzerlegung jedes mal für jede Augmentierung -> Lösung: zusätzlicher Parameter 'aug_count' für Methode fancy_pca.py ###

### TODO: Amir fragen: fancy_pca.py, Zeile 21-22: richtiges alpha nach Paper (mit Olli erstellt)? Es erschient so, siehe Standardfehler auf 5 setzen ###

class FancyPCA: 
    
 
    def fancy_pca(self, data):
        mean_free_data = data - np.mean(data, axis=(0,1)) # Berechnung des Mittelwerts der Daten entlang der ersten beiden Achsen und Subtraktion, um die Daten zu zentrieren
        data_covarianz = np.cov(mean_free_data, rowvar=False) # Berechnung der Kovarianzmatrix der zentrierten Daten; rowvar=False: jede Spalte ist eine Variable
        eig_values, eig_vecs = np.linalg.eigh(data_covarianz) # Berechnung der Eigenwerte und Eigenvektoren der Kovarianzmatrix
        #print('test', eig_values / np.sum(eig_values)) ### Debugging ###
        # Zufällige Skalierung der Eigenwerte mit einer Normalverteilung basierend auf Mittelwert und Standardabweichung
        alpha = constants.FANCY_PCA_STANDARD_DEVIATION * np.random.randn(3) + constants.FANCY_PCA_MEAN
        print(alpha)
        #alpha = np.random.normal(self.mean, self.standard_deviation) # Pauls alpha (ist nur eine Konstante, aber wir brauchen alpha_i, i=1,2,3)
        #print(alpha) ### Debugging ###
        eig_values *= alpha
        #print('test2', eig_values) ### Debugging ###

        p_mat = np.array(eig_vecs) # Umwandlung der Eigenvektoren in eine Matrixform
        add_vec = np.dot(p_mat, eig_values) # Berechnung des Modifikationsvektors durch Multiplikation von Matrix (bestehend aus Eigenvektoren) und Vektor (bestehend aus Eigenwerten)
        data += add_vec # Addition des Modifikationsvektors zu den Originaldaten
        data = np.clip(data, 0, 1) # Begrenzung der Werte im Datenarray auf den Bereich [0, 1], um unerwünschte Werte zu vermeiden
        return data # Rückgabe der transformierten Daten