### WICHTIG: vor einem erneuten Run dieser Datei: den Ordner ./out_fancy_pca_caltech101 löschen ###

import os
import numpy as np
import constants
import fancy_pca as FP
from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
from PIL import Image
import ssl
import parser

# SSL fix für HTTPS-Downloads (notwendig bei Caltech101)
ssl._create_default_https_context = ssl._create_unverified_context

# Zielverzeichnis für Ausgabe
output_path = './out_fancy_pca_caltech101_std_0'
os.makedirs(output_path, exist_ok=True)

# Caltech101 Dataset laden mit Transformation
caltech_dataset = Caltech101(
    root='./data_caltech101',
    download=True,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # gibt Werte zwischen 0 und 1 zurück
    ])
)

print(f"Anzahl der Bilder im Dataset: {len(caltech_dataset)}")

# Datensatz in Trainings- und Testdaten aufteilen
train_data, test_data = random_split(caltech_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# DataLoader für das Training
data_loader = DataLoader(train_data, batch_size=1, shuffle=False)

# FancyPCA-Instanz
fancy_pca_transform = FP.FancyPCA()

# Iteriere über das Dataset und wende Fancy PCA an
for data_index, (image_tensor, label) in enumerate(data_loader):
    for i in range(len(image_tensor)): # In diesem Fall wird len(image_tensor) immer 1 sein (batch_size=1)
        # Erstelle Ausgabeverzeichnis für die aktuelle Klasse, z. B. ./out_fancy_pca_caltech101/42
        os.makedirs(os.path.join(output_path, str(label[i].item())), exist_ok=True)

        # (1, 3, 224, 224) → (224, 224, 3)
        # Überprüfe, ob der Tensor 2D oder 3D ist
        image_np = image_tensor[i].squeeze(0)  # Entferne die Batch-Dimension (falls vorhanden)

        # Wenn der Tensor nur 2D ist (also kein Kanal), füge einen Kanal hinzu
        if len(image_np.shape) == 2:  # Wenn das Bild nur HxW hat (ohne Kanal)
            image_np = image_np.unsqueeze(0).repeat(3, 1, 1)  # Füge 3 Kanäle hinzu (RGB)

        # Jetzt hat der Tensor die Form (3, H, W), du kannst permute verwenden
        image_np = image_np.permute(1, 2, 0).numpy()  # Umwandeln zu (H, W, 3)

        height, width, _ = image_np.shape
        image_name = f"caltech_img_{data_index}" # Generiere Bildnamen wie caltech_img_0, caltech_img_1, ...

        # Bild vorbereiten, wie in parser.parse()
        data_array = np.vstack(image_np * constants.MAX_COLOR_VALUE) / constants.MAX_COLOR_VALUE
        # Speichert Bildgröße als Liste von Tuples – FancyPCA erwartet das Format
        size_images = [(width, height)]

        # Augmentierungsschleife
        for aug_count in range(constants.AUG_COUNT):
            print(f"Fancy_PCA: {image_name}, {aug_count + 1}/{constants.AUG_COUNT}")
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array)

            # Augmentiertes Bild speichern im Ausgabeverzeichnis, in Ordner nach Klasse sortiert
            parser.save_data(
                fancy_pca_images,
                image_name + f"_{aug_count + 1}",
                size_images,
                aug_count,
                0,  # index 0, weil size_images nur einen Eintrag hat
                path=output_path+"/" + str(label[i].item()) # Zielpfad, z. B. ./out_fancy_pca_caltech101/42
            )