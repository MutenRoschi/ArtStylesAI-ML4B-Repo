import streamlit as st
import torch
import gdown
from PIL import Image
from torchvision import transforms
from io import BytesIO

# Funktion zum Laden des Modells
def load_model_from_drive():
    # Google Drive Link zum Modell
    url = 'https://drive.google.com/uc?export=download&id=1tlUGX7lDPnCohK8Q_pqG47ALYbeS468g'
    output = 'latest_net_G_A.pth'
    gdown.download(url, output, quiet=False)

    # Modell laden
    model = torch.load(output)
    model.eval()  # Modell in den Evaluierungsmodus versetzen
    return model

# Lade das CycleGAN Modell
model = load_model_from_drive()

def transform_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image).unsqueeze(0)  # Bild transformieren und Batch-Dimension hinzufügen
    transformed_image = model(image)  # Bild transformieren
    return transformed_image

def main():
    st.title("Baroque to Realism - CycleGAN Style Transfer")

    # Lade das Bild
    uploaded_file = st.file_uploader("Wähle ein Bild", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Eingabebild", use_column_width=True)

        # Transformiere das Bild
        transformed_image = transform_image(image)
        st.image(transformed_image, caption="Generiertes Bild", use_column_width=True)

if __name__ == "__main__":
    main()
