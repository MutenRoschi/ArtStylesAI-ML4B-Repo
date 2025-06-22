import streamlit as st
import torch
import gdown
from torchvision import transforms
from PIL import Image
import os
from torch import nn
import tempfile

# Lade das CycleGAN-Modell
class CycleGANModel(nn.Module):
    def __init__(self):
        super(CycleGANModel, self).__init__()
        # Hier würdest du dein CycleGAN-Modell implementieren oder laden
        # Z.B. Selbstnetzwerk-Definitionen, wie Generatoren (netG), Diskriminatoren (netD), etc.
        self.netG = self.load_generator_model()

    def forward(self, input_image):
        # Vorhersage und Umwandlung des Bildes
        return self.netG(input_image)  # Das transformierte Bild wird hier zurückgegeben

    def load_generator_model(self):
        # Lade den Generator des Modells, hier kannst du die Architektur von CycleGAN nachbilden
        pass  # Deine Architektur hier einfügen

# Lade das Modell von Google Drive
def load_model_from_drive():
    # Google Drive Link zum Modell
    url = 'https://drive.google.com/uc?export=download&id=1tlUGX7lDPnCohK8Q_pqG47ALYbeS468g'  # Dein Link zum Modell
    output = 'latest_net_G_A.pth'

    # Lade das Modell
    print("Downloading model...")
    gdown.download(url, output, quiet=False)  # Modell herunterladen

    # Erstelle das CycleGAN-Modell
    model = CycleGANModel()

    # Lade den Zustand (state_dict) des Modells
    checkpoint = torch.load(output)  # Hier wird die Checkpoint-Datei geladen
    model.load_state_dict(checkpoint)  # Zustand des Modells laden

    model.eval()  # Setze das Modell in den Evaluierungsmodus
    return model

# Bild transformieren
def transform_image(uploaded_image, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)  # Füge eine Batch-Dimension hinzu
    transformed_image = model(image)  # Transformiere das Bild
    return transformed_image

# Hauptfunktion für die Streamlit-App
def main():
    st.title("Baroque to Realism - CycleGAN Style Transfer")
    
    # Lade das CycleGAN-Modell
    model = load_model_from_drive()

    # Lade das Bild vom Nutzer
    uploaded_file = st.file_uploader("Wähle ein Bild", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Lade das Bild
        image = Image.open(uploaded_file)
        st.image(image, caption="Eingabebild", use_column_width=True)

        # Transformiere das Bild
        transformed_image = transform_image(image, model)
        # Konvertiere das transformierte Bild zurück in ein Bild
        transformed_image_pil = transforms.ToPILImage()(transformed_image.squeeze(0))
        st.image(transformed_image_pil, caption="Generiertes Bild", use_column_width=True)

# Starte die Streamlit-App
if __name__ == "__main__":
    main()
