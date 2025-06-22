import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Verknüpfung mit dem öffentlichen Google Drive Ordner "ML4B-CycleGAN"
model_url = "https://drive.google.com/drive/folders/1rTzT78v7ssT4RZu1H1Lm36b3FRThT19Q?usp=sharing" 

# Lade das Modell herunter
def download_file_from_drive(url):
    response = requests.get(url)
    return response.content

# Lade das CycleGAN-Modell
class CycleGANModel(torch.nn.Module):
    def __init__(self):
        super(CycleGANModel, self).__init__()

    def forward(self, input_image):
        # Vorhersage und Umwandlung des Bildes
        return transformed_image

# Laden des Modells
model_data = download_file_from_drive(model_url)
checkpoint = torch.load(BytesIO(model_data))
model = CycleGANModel()
model.load_state_dict(checkpoint)

# Bildtransformationsfunktion
def transform_image(uploaded_image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = transform(uploaded_image)
    image = image.unsqueeze(0)  
    transformed_image = model(image)
    return transformed_image

def main():
    st.title("Baroque to Realism - CycleGAN Style Transfer")
    
    uploaded_file = st.file_uploader("Wähle ein Bild", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Eingabebild", use_column_width=True)

        # Transformiere das Bild
        transformed_image = transform_image(image)
        st.image(transformed_image, caption="Generiertes Bild", use_column_width=True)

if __name__ == "__main__":
    main()

