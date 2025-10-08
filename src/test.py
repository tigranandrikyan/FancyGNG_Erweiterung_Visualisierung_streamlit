import os  # Modul für Betriebssystem-Funktionen wie Pfadoperationen und Ordnererstellung
import numpy as np  # NumPy für effiziente numerische Berechnungen mit Arrays
import constants  # Benutzerdefinierte Datei mit Konstanten wie MAX_COLOR_VALUE, AUG_COUNT
import fancy_pca as FP  # Eigenes Modul mit einer FancyPCA-Klasse für Datenaugmentation
from torchvision.datasets import Caltech101  # Importiert das Caltech101-Dataset über torchvision
from torchvision import transforms  # Transformationen für Bildvorverarbeitung (z. B. Resize, ToTensor)
from torch.utils.data import DataLoader, random_split  # DataLoader für Batch-Verarbeitung, random_split für Trainings/Test-Split
import torch  # PyTorch-Framework für Deep Learning und Tensor-Operationen
from PIL import Image  # Pillow für Bildverarbeitung (nicht direkt genutzt in diesem Skript)
import ssl  # SSL-Modul zur Konfiguration von HTTPS-Downloads
import parser  # Benutzerdefiniertes Modul, vermutlich zum Speichern der augmentierten Bilder

# Deaktiviert SSL-Zertifikatüberprüfung (z. B. wenn es Probleme beim Download von Caltech101 gibt)
ssl._create_default_https_context = ssl._create_unverified_context

# Pfad zum Speicherort der ausgegebenen Bilder
output_path = './out_fancy_pca_caltech101'
os.makedirs(output_path, exist_ok=True)  # Erstellt den Ordner, wenn er nicht existiert

# Caltech101-Datensatz laden und vorbereiten (Bilder auf 224x224 skalieren und in Tensoren umwandeln)
caltech_dataset = Caltech101(
    root='./data_caltech101',  # Speicherort für das heruntergeladene Dataset
    download=True,  # Dataset herunterladen, wenn nicht vorhanden
    transform=transforms.Compose([
        transforms.Resize((224, 224)),  # Bilder auf 224x224 Pixel skalieren
        transforms.ToTensor(),  # Bilder in Tensoren mit Werten von 0 bis 1 umwandeln
    ])
)

# Gibt die Gesamtanzahl der Bilder im Datensatz aus
print(f"Anzahl der Bilder im Dataset: {len(caltech_dataset)}")

# Aufteilen des Datensatzes in Trainings- (80 %) und Testdaten (20 %) mit festem Seed für Reproduzierbarkeit
train_data, test_data = random_split(caltech_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# DataLoader für das Training – lädt jeweils ein Bild (batch_size=1) in der ursprünglichen Reihenfolge (kein shuffle)
data_loader = DataLoader(train_data, batch_size=1, shuffle=False)

# Erstellt eine Instanz der FancyPCA-Klasse für spätere Bildaugmentation
fancy_pca_transform = FP.FancyPCA()

# Iteration über alle Trainingsbilder mit ihren Labels
for data_index, (image_tensor, label) in enumerate(data_loader):
    for i in range(len(image_tensor)):  # In diesem Fall wird len(image_tensor) immer 1 sein (batch_size=1)
        # Erstelle Ausgabeverzeichnis für die aktuelle Klasse, z. B. ./out_fancy_pca_caltech101/42
        os.makedirs(os.path.join(output_path, str(label[i].item())), exist_ok=True)

        # Entfernt die Batch-Dimension (von [1, 3, 224, 224] auf [3, 224, 224])
        image_np = image_tensor[i].squeeze(0)

        # Falls das Bild nur zwei Dimensionen hat (H x W), füge RGB-Kanäle hinzu (macht aus Graustufen ein RGB-Bild)
        if len(image_np.shape) == 2:
            image_np = image_np.unsqueeze(0).repeat(3, 1, 1)

        # Ändert Tensor von (C, H, W) zu (H, W, C) und wandelt ihn in NumPy-Array um
        image_np = image_np.permute(1, 2, 0).numpy()

        # Extrahiere Höhe und Breite des Bildes
        height, width, _ = image_np.shape
        image_name = f"caltech_img_{data_index}"  # Generiere Bildnamen wie caltech_img_0, caltech_img_1, ...

        # Skaliere Bildwerte zurück auf 0–255 und staple sie als flaches Array (für FancyPCA erwartet)
        data_array = np.vstack(image_np * constants.MAX_COLOR_VALUE) / constants.MAX_COLOR_VALUE

        # Speichert Bildgröße als Liste von Tuples – FancyPCA erwartet das Format
        size_images = [(width, height)]

        # Schleife über die Anzahl der gewünschten Augmentierungen
        for aug_count in range(constants.AUG_COUNT):
            print(f"Fancy_PCA: {image_name}, {aug_count + 1}/{constants.AUG_COUNT}")

            # Augmentiertes Bild erzeugen mit FancyPCA
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array)

            # Augmentiertes Bild speichern im Ausgabeverzeichnis, in Ordner nach Klasse sortiert
            parser.save_data(
                fancy_pca_images,  # augmentiertes Bild
                image_name + f"_{aug_count + 1}",  # z. B. caltech_img_0_1
                size_images,  # Bildgröße, z. B. (224, 224)
                aug_count,  # Augmentierungs-Index
                0,  # Index 0, da size_images nur ein Bild enthält
                path=output_path + "/" + str(label[i].item())  # Zielpfad, z. B. ./out_fancy_pca_caltech101/42
            )
