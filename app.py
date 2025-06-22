import streamlit as st
import torch
import gdown
from torchvision import transforms
from PIL import Image
import os
from torch import nn

# Lade das CycleGAN-Modell
class CycleGANModel(nn.Module):
    def __init__(self):
        super(CycleGANModel, self).__init__()
        # Hier würdest du dein CycleGAN-Modell implementieren oder laden
        # z.B. self.netG = ...

    def forward(self, input_image):
        # Vorhersage und Umwandlung des Bildes
        return transformed_image  # Hier das Bild zurückgeben

def load_model_from_drive():
    # Google Drive Link zum Modell
    url = 'https://drive.google.com/uc?export=download&id=1tlUGX7lDPnCohK8Q_pqG47ALYbeS468g'
    output = 'latest_net_G_A.pth'
    print("Downloading model...")
    gdown.download(url, output, quiet=False)  # Modell herunterladen

    # Erstelle das CycleGAN-Modell
    model = CycleGANModel()

    # Lade den Zustand (state_dict) des Modells
    checkpoint = torch.load(output)
    model.load_state_dict(checkpoint)  # Zustand des Modells laden

    model.eval()  # Das Modell in den Evaluierungsmodus versetzen
    return model

def transform_image(uploaded_image, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)  # Batch-Dimension hinzufügen
    transformed_image = model(image)  # Das Bild transformieren
    return transformed_image

def main():
    st.title("Baroque to Realism - CycleGAN Style Transfer")
    
    # Lade das Modell
    model = load_model_from_drive()

    # Lade das Bild
    uploaded_file = st.file_uploader("Wähle ein Bild", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Lade das Bild
        image = Image.open(uploaded_file)
        st.image(image, caption="Eingabebild", use_column_width=True)

        # Transformiere das Bild
        transformed_image = transform_image(image, model)
        st.image(transformed_image, caption="Generiertes Bild", use_column_width=True)

if __name__ == "__main__":
    main()
