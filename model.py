import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("eurosat_resnet18.pth", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5095, 0.4705, 0.4593], std=[0.2341, 0.1267, 0.1015])
])

st.title(" Satellite Image Classifier")
st.markdown("Upload a satellite image and I'll classify it for you!")
st.markdown(" Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,Pasture, PermanentCrop, Residential, River, SeaLake")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = probs.argmax().item()
        pred_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

    st.success(f"Prediction: **{pred_class}** ({confidence*100:.2f}% confidence)")

