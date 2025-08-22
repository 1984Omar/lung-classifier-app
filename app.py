# app.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import traceback

# =============================================================================
# Configuration
# =============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ["Benign", "Malignant", "Normal"]
DEFAULT_IMG_SIZE = 224
MODEL_PATH = "model.pth"

# Use CPU, as it's standard for Streamlit Cloud's free tier
device = torch.device("cpu")

# Image transformations (must match the training pipeline)
data_transforms = transforms.Compose([
    transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =============================================================================
# Model Definition (Your Original Code)
# =============================================================================
class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=1, padding=0)

    def forward(self, x):
        channel_pooled = F.adaptive_avg_pool2d(x, (1, 1))
        channel_features = F.relu(self.conv1(channel_pooled))
        channel_attention = torch.sigmoid(self.conv2(channel_features))
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = torch.sigmoid(self.conv3(spatial_features))
        attention_out = x * channel_attention * spatial_attention
        return x + attention_out, spatial_attention

class MedicalImageClassifier(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet_v2", pretrained=False):
        super(MedicalImageClassifier, self).__init__()
        self.backbone_name = backbone
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        if backbone == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights=weights)
            self.feature_dim = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()
            self.attention_layer = AttentionBlock(self.feature_dim)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.attention_maps = None

    def forward(self, x):
        features = self.backbone.features(x)
        features_attn, attention_map = self.attention_layer(features)
        self.attention_maps = attention_map
        x_out = self.gap(features_attn)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.fc2(x_out)
        return x_out, features_attn

# =============================================================================
# Helper Functions for Streamlit
# =============================================================================

@st.cache_resource
def load_model(model_path, num_classes):
    """Loads the model using Streamlit's caching to prevent reloading."""
    model = MedicalImageClassifier(num_classes=num_classes, backbone="mobilenet_v2", pretrained=False)
    try:
        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure it's in the root of your repository.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        traceback.print_exc()
        return None

def preprocess_image(image_bytes):
    """Preprocesses an image from bytes for model prediction."""
    try:
        image = Image.open(image_bytes).convert('RGB')
        input_tensor = data_transforms(image).unsqueeze(0)
        return input_tensor.to(device), image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def predict(model, input_tensor):
    """Runs model inference."""
    try:
        with torch.no_grad():
            outputs, _ = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probabilities).item()
            pred_name = CLASS_NAMES[pred_idx]
        return pred_name, probabilities.cpu().numpy(), pred_idx
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

def get_target_layer(model):
    """Finds the target layer for Grad-CAM visualization."""
    return model.backbone.features[-1]

def generate_gradcam_overlay(model, input_tensor, original_image, target_layer, target_class_index):
    """Generates Grad-CAM++ visualization."""
    if target_layer is None:
        return np.array(original_image)
    targets = [ClassifierOutputTarget(target_class_index)]
    try:
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0, :]
            original_np = np.array(original_image) / 255.0
            visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
            return visualization
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return np.array(original_image)

# =============================================================================
# Streamlit User Interface
# =============================================================================

st.set_page_config(page_title="Lung Image Classifier", layout="wide")

st.title("🩺 Lung Medical Image Classifier")
st.markdown("##### By Omar Ibrahim Obaid")
st.markdown("""
This web app utilizes a deep learning model to classify lung CT scan images as **Benign**, **Malignant**, or **Normal**.
Upload an image to receive a prediction and a Grad-CAM++ visualization that highlights the areas the model focused on.
""")

# --- Load the model ---
model = load_model(MODEL_PATH, NUM_CLASSES)

if model:
    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Upload a lung CT image...",
        type=["png", "jpg", "jpeg", "bmp", "tif"]
    )

    if uploaded_file is not None:
        st.header("Analysis Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            input_tensor, original_image = preprocess_image(uploaded_file)
            st.image(original_image, caption="Your uploaded image.", use_column_width=True)

        with col2:
            st.subheader("Prediction")
            if input_tensor is not None:
                pred_name, probs, pred_idx = predict(model, input_tensor)
                
                # Display colored prediction text
                color_map = {"Malignant": "red", "Benign": "green", "Normal": "blue"}
                st.markdown(f"### <span style='color:{color_map.get(pred_name, 'white')};'>Prediction: {pred_name}</span>", unsafe_allow_html=True)

                st.write("#### Probabilities:")
                for name, prob in zip(CLASS_NAMES, probs):
                    st.write(f"- **{name}**: {prob:.4f}")

                st.write("---")
                st.subheader("Grad-CAM++ Visualization")
                st.info("This heatmap shows which parts of the image were most influential for the prediction.")
                
                target_layer = get_target_layer(model)
                if target_layer:
                    with st.spinner("Generating visualization..."):
                        gradcam_overlay = generate_gradcam_overlay(model, input_tensor, original_image, target_layer, pred_idx)
                        st.image(gradcam_overlay, caption=f"Grad-CAM++ for '{pred_name}'", use_column_width=True)
                else:
                    st.warning("Could not identify a target layer for Grad-CAM.")
else:
    st.error("The application cannot start because the model failed to load.")

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    f"""
    **Model Architecture:**
    - **Backbone:** MobileNetV2
    - **Custom Layer:** Attention Block
    - **Classes:** {', '.join(CLASS_NAMES)}
    - **Inference Device:** {str(device).upper()}
    """
)
st.sidebar.markdown("Created by **Omar Ibrahim Obaid**.")