import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import requests
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie

# Function to load Lottie animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# App title and introduction
st.title("🔍 Product Quality Inspection System")
st.write("This is a simple web app to upload images and check for product matches.")
st.write("👨‍💻 Developed by **Tathagat Shaw**, Data Scientist & ML Engineer.")

# Load and display animation
lottie_animation = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json")
if lottie_animation:
    st_lottie(lottie_animation, height=300, key="intro_animation")
st.markdown("<hr>", unsafe_allow_html=True)

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Feature extraction function
def extract_features(img, model):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Upload reference images
st.subheader("📂 Upload Reference Product Images")
reference_images = st.file_uploader(
    "Upload multiple reference images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

reference_features = []
reference_image_files = []

if reference_images:
    for img_file in reference_images:
        img = Image.open(img_file).convert("RGB")
        features = extract_features(img, model)
        reference_features.append(features)
        reference_image_files.append(img_file)
    reference_features = np.array(reference_features)

st.markdown("<hr>", unsafe_allow_html=True)

# Upload new image
st.subheader("📸 Upload a New Image to Verify")
new_image_file = st.file_uploader("Upload image to verify", type=["jpg", "jpeg", "png"])

# Product match function
def check_product_match(new_image):
    new_image_features = extract_features(new_image, model)
    similarities = cosine_similarity([new_image_features], reference_features)
    max_index = np.argmax(similarities)
    max_similarity = similarities[0][max_index]
    closest_match = reference_image_files[max_index]
    return max_similarity > 0.8, closest_match

# Check and display result
if new_image_file and len(reference_features) > 0:
    new_image = Image.open(new_image_file).convert("RGB")
    is_match, closest_match = check_product_match(new_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(new_image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(Image.open(closest_match), caption="Closest Match", use_container_width=True)

    if is_match:
        st.success("✅ The uploaded product matches the reference product.")
    else:
        st.error("❌ The uploaded product does NOT match the reference product.")