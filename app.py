import torch
from torch import nn
import gdown

# Lade das CycleGAN-Modell
class CycleGANModel(nn.Module):
    def __init__(self):
        super(CycleGANModel, self).__init__()
        # Hier würdest du dein CycleGAN-Modell laden, z.B.:
        # self.netG = load_generator_model()

    def forward(self, input_image):
        # Vorhersage und Umwandlung des Bildes
        return transformed_image  # zurückgegebenes Bild


# Laden des Modells von einer URL
def load_model_from_drive():
    url = 'https://drive.google.com/uc?export=download&id=1tlUGX7lDPnCohK8Q_pqG47ALYbeS468g'  # Google Drive URL
    output = 'latest_net_G_A.pth'
    gdown.download(url, output, quiet=False)  # Download der Modell-Datei
    model = CycleGANModel()
    checkpoint = torch.load(output)
    model.load_state_dict(checkpoint)
    return model

# Lade das Modell
model = load_model_from_drive()

def transform_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)  # Batch-Dimension hinzufügen
    transformed_image = model(image)
    return transformed_image

# Streamlit-App
import streamlit as st
from PIL import Image

def main():
    st.title("Baroque to Realism - CycleGAN Style Transfer")
    
    uploaded_file = st.file_uploader("Wähle ein Bild", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Lade das Bild
        image = Image.open(uploaded_file)
        st.image(image, caption="Eingabebild", use_column_width=True)

        # Transformiere das Bild
        transformed_image = transform_image(image)
        st.image(transformed_image, caption="Generiertes Bild", use_column_width=True)

if __name__ == "__main__":
    main()
