import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image


@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)   
    model.classifier[1] = nn.Linear(1280, 4)  
    model.load_state_dict(torch.load("modelo_cancer.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

CLASSES = model_classes = ['Benign', '[Malignant] Pre-B','[Malignant] Pro-B','[Malignant] early Pre-B']


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)
    return CLASSES[pred.item()]

st.title("🔬 Classificação de Células Cancerígenas")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    if st.button("Classificar"):
        with st.spinner("Processando..."):
            r = predict(image)
        st.success(f"Classe identificada: **{r}**")
