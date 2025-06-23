import os
import gdown
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# 1. Generator-Architektur aus de CycleGAN-Repo 
from models.networks import define_G      # liegt jetzt in unserem Repo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource   # nur 1× pro Session laden
def load_generator():
    # Google-Drive-Link der .pth-Datei (beinhaltet die Gewichtungen, wichtig!)
    url = "https://drive.google.com/uc?id=1tlUGX7lDPnCohK8Q_pqG47ALYbeS468g"
    ckpt_path = "latest_net_G_A.pth"
    if not os.path.exists(ckpt_path):
        gdown.download(url, ckpt_path, quiet=False)

    # Architektur exakt wie im Training:
    netG = define_G(
        input_nc=3, output_nc=3, ngf=64,
        netG="resnet_9blocks",
        norm="instance", use_dropout=False,
        init_type="normal", init_gain=0.02, gpu_ids=[]
    )

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    netG.load_state_dict(state_dict)       
    netG.to(DEVICE).eval()
    return netG

GEN = load_generator()

# 2. Hilfsfunktionen 
TRANSFORM_IN = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def stylize(pil_img):
    x = TRANSFORM_IN(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = GEN(x)[0]
    # Rück­trans­form für Anzeige
    y = (y * 0.5 + 0.5).clamp(0, 1).cpu()
    return transforms.ToPILImage()(y)

# 3. Streamlit-UI 
st.title("🎨 Baroque → Realism Style Transfer (CycleGAN)")

uploaded = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_column_width=True)

    st.write("⏳ Style-Transfer läuft …")
    out_img = stylize(img)
    st.image(out_img, caption="Transformiert", use_column_width=True)
