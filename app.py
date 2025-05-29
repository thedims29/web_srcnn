import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import math

# Load model SRCNN saja
model_srcnn = load_model('srcnn_model.keras', compile=False)

# Metrics calculation
def calculate_metrics(original, restored):
    original = np.clip(original, 0, 1)
    restored = np.clip(restored, 0, 1)
    mse_val = mean_squared_error(original, restored)
    rmse_val = math.sqrt(mse_val)
    psnr_val = peak_signal_noise_ratio(original, restored, data_range=1.0)
    try:
        ssim_val = structural_similarity(original, restored, channel_axis=-1, data_range=1.0)
    except ValueError:
        ssim_val = 0.0  # fallback jika ukuran terlalu kecil
    return mse_val, rmse_val, psnr_val, ssim_val

# Apply blur to simulate damage
def add_blur(image, blur_level):
    if blur_level == 0:
        return image
    return cv2.GaussianBlur(image, (2 * blur_level + 1, 2 * blur_level + 1), 0)

# Predict function
def predict_image(model, image):
    input_img = image.astype(np.float32)
    pred = model.predict(np.expand_dims(input_img, axis=0), verbose=0)
    return np.clip(pred[0], 0.0, 1.0)

# Streamlit UI
st.set_page_config(layout="wide", page_title="Restorasi Citra")
st.title("🖼️ Restorasi Citra dengan SRCNN")

# UI Layout
col1, col2, col3 = st.columns([1.2, 1.2, 1.2])

with col1:
    uploaded_file = st.file_uploader("Upload Citra", type=["jpg", "png", "jpeg"])
    blur_level = st.slider("Tingkat Blur", min_value=0, max_value=10, value=5)
    if st.button("Proses"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((256, 256))
            image_np = np.array(image) / 255.0

            # Blur image
            blurred_image = add_blur((image_np * 255).astype(np.uint8), blur_level)
            blurred_image = blurred_image.astype(np.float32) / 255.0

            # Predict with SRCNN
            output_srcnn = predict_image(model_srcnn, blurred_image)

            # Save to session
            st.session_state['original'] = image_np
            st.session_state['blurred'] = blurred_image
            st.session_state['srcnn'] = output_srcnn

# Show results
if 'original' in st.session_state:
    col1.image(st.session_state['blurred'], caption="Citra Blur", use_container_width=True)
    col2.image(st.session_state['original'], caption="Before", use_container_width=True)
    col3.image(st.session_state['srcnn'], caption="SRCNN", use_container_width=True)

    # Metric Display Function
    def render_metrics(col, title, target):
        mse, rmse, psnr_val, ssim_val = calculate_metrics(st.session_state['original'], target)
        with col:
            st.markdown(f"**Parameter**  \n**{title}**")
            st.markdown(f"MSE  : {mse:.4f}")
            st.markdown(f"RMSE : {rmse:.4f}")
            st.markdown(f"PSNR : {psnr_val:.2f}")
            st.markdown(f"SSIM : {ssim_val:.4f}")

    st.markdown("---")
    col4, col5 = st.columns(2)
    render_metrics(col4, "Before", st.session_state['blurred'])
    render_metrics(col5, "SRCNN", st.session_state['srcnn'])
