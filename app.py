
import streamlit as st
import os
import torch
import torch.nn as nn
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image

# Hide extra warnings or stack traces
st.set_option("client.showErrorDetails", False)

st.title("Chest X-Ray Pneumonia Detection")
st.write("##### Upload a chest X-ray image in (PNG, JPG, or JPEG) fromat and the AI-Powered model will predict if person has pneumonia or not")


# 1. Hard-coded Model Path
MODEL_PATH = "D:/Talha_Folder/py-learning/best_pneumonia_model.pth"

@st.cache_resource
def load_model(model_path: str):
    """Load the DenseNet model from TorchXRayVision with a custom classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.op_threshs = None
    # Classifier with dropout, exactly as used during training
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load the model, or show an error if not found
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"**Model file not found:**\n\n`{MODEL_PATH}`\n\n"
        "Please verify that the file path is correct."
    )
    st.stop()

# 2. Preprocessing Transform (same as training normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
st.write("")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Preprocess for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        probability = torch.sigmoid(output).item()
        pred_class = "PNEUMONIA" if probability > 0.5 else "NORMAL"

    # Denormalize image for display
    image_np = input_image.cpu().squeeze().numpy()
    image_denorm = (image_np * 0.5) + 0.5  # back to [0,1]

    # 3. Display images side by side with extra spacing
    #    We'll create three columns: left image, empty space, right image
    colA, colSpace, colB = st.columns([2, 1, 2], gap="large")

    with colA:
        st.image(image, caption="Uploaded Chest X-Ray image", width=400)

    with colSpace:
        # Just an empty column to create spacing
        st.write("")  # or pass

    with colB:
        st.image(image_denorm, caption="Zoomed-in Chest X-Ray Image", width=400)

    # 4. Show Diagnosis (no probability)
    st.write("## Diagnosis")
    if pred_class == "PNEUMONIA":
        st.error(f"**{pred_class}**")
    else:
        st.success(f"**{pred_class}**")
