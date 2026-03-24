import streamlit as st
import cv2
import time
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & CACHING
# ==============================================================================
st.set_page_config(page_title="Sneaker Authenticator", page_icon="👟", layout="wide")

import os
# This automatically gets the folder where app.py is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The rest stays exactly the same!
YOLO_SHOE_PATH = os.path.join(BASE_DIR, "shoe_best.pt")
YOLO_MICRO_PATH = os.path.join(BASE_DIR, "micro_best.pt")
TOWER_PATH = os.path.join(BASE_DIR, "siamese_tower.keras")
MLP_PATH = os.path.join(BASE_DIR, "fusion_mlp.keras")
BANK_PATH = os.path.join(BASE_DIR, "reference_bank.pkl")

@st.cache_resource
def load_models():
    """Loads all 4 models into memory once to prevent lag on every button click."""
    yolo_shoe = YOLO(YOLO_SHOE_PATH)
    yolo_micro = YOLO(YOLO_MICRO_PATH)
    tower = tf.keras.models.load_model(TOWER_PATH, compile=False)
    mlp = tf.keras.models.load_model(MLP_PATH, compile=False)
    return yolo_shoe, yolo_micro, tower, mlp

@st.cache_data
def load_reference_bank():
    with open(BANK_PATH, 'rb') as f:
        return pickle.load(f)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def crop_box(img, box):
    if box is None: return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]

def resize_norm(img, size=(128, 128)):
    if img is None or img.size == 0: return np.zeros(size + (3,), dtype=np.float32)
    return cv2.resize(img, size).astype(np.float32) / 255.0

def compute_stats_fast(emb, bank_embs):
    if bank_embs is None or len(bank_embs) == 0: return [1.0, 1.0]
    sims = np.dot(bank_embs, emb)
    dists = 1.0 - sims
    return [float(np.min(dists)), float(np.mean(dists))]

# ==============================================================================
# UI LAYOUT & LOGIC
# ==============================================================================
st.title("👟 Deep CNN-Siamese Sneaker Authenticator")
st.markdown("**Master's Thesis Prototype** | *Upload an image of a Nike Air Force 1 or Jordan 1 to analyze its micro-features.*")

# Load assets silently in the background
with st.spinner("Loading Deep Learning Models..."):
    yolo_shoe, yolo_micro, tower, mlp = load_models()
    ref_bank = load_reference_bank()

uploaded_file = st.file_uploader("Upload Sneaker Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Authenticate Sneaker", type="primary"):
        start_time = time.perf_counter()
        
        with st.spinner("Analyzing Whole Shoe..."):
            res_shoe = yolo_shoe(img_array, conf=0.25, verbose=False)[0]
            
            if len(res_shoe.boxes) == 0:
                st.error("No shoe detected in the image. Please try another angle.")
            else:
                best_idx = int(np.argmax(res_shoe.boxes.conf.cpu().numpy()))
                shoe_box = res_shoe.boxes.xyxy.cpu().numpy()[best_idx]
                shoe_conf = float(res_shoe.boxes.conf.cpu().numpy()[best_idx])
                shoe_cls = int(res_shoe.boxes.cls.cpu().numpy()[best_idx])
                
                shoe_crop = crop_box(img_array, shoe_box)
                
                with st.spinner("Extracting Micro-Features..."):
                    res_micro = yolo_micro(shoe_crop, verbose=False)[0]
                    logo_box, stitch_box, max_l, max_s = None, None, -1, -1
                    
                    if len(res_micro.boxes) > 0:
                        confs = res_micro.boxes.conf.cpu().numpy()
                        boxes = res_micro.boxes.xyxy.cpu().numpy()
                        clss = res_micro.boxes.cls.cpu().numpy().astype(int)
                        for i, c in enumerate(clss):
                            if c == 0 and confs[i] > max_l: logo_box, max_l = boxes[i], confs[i]
                            elif c == 1 and confs[i] > max_s: stitch_box, max_s = boxes[i], confs[i]
                            
                    logo_crop = crop_box(shoe_crop, logo_box)
                    stitch_crop = crop_box(shoe_crop, stitch_box)
                    
                    logo_img = resize_norm(logo_crop)
                    stitch_img = resize_norm(stitch_crop)
                    
                    # Siamese Embeddings
                    batch_imgs = np.stack([logo_img, stitch_img])
                    embeddings = tower(batch_imgs, training=False).numpy()
                    logo_emb, stitch_emb = embeddings[0], embeddings[1]
                    
                    # Distances
                    l_stats = compute_stats_fast(logo_emb, ref_bank.get("ShoeLogo", []))
                    s_stats = compute_stats_fast(stitch_emb, ref_bank.get("StitchingPatterns", []))
                    
                    # Feature Fusion
                    shoe_h, shoe_w = shoe_crop.shape[:2]
                    img_h, img_w = img_array.shape[:2]
                    pred_one_hot = np.zeros(4, dtype=np.float32)
                    if 0 <= shoe_cls < 4: pred_one_hot[shoe_cls] = 1.0
                    
                    fvec = np.concatenate([
                        np.array(l_stats + s_stats, dtype=np.float32), 
                        np.array([shoe_conf, (shoe_h*shoe_w)/max(img_h*img_w, 1), shoe_w/max(shoe_h, 1)], dtype=np.float32),
                        pred_one_hot, logo_emb, stitch_emb
                    ])
                    
                    # Final Classification
                    prob_original = float(mlp(np.expand_dims(fvec, axis=0), training=False).numpy()[0][0])
                    
        # Calculate Latency
        exec_time = (time.perf_counter() - start_time) * 1000
        
        # --- UI RESULTS DISPLAY ---
        st.markdown("---")
        st.subheader("Analysis Results")
        
        cols = st.columns(3)
        with cols[0]:
            st.image(shoe_crop, caption="Detected Shoe")
        with cols[1]:
            if logo_crop is not None: st.image(logo_crop, caption=f"Extracted Logo (Dist: {l_stats[0]:.2f})")
            else: st.warning("Logo not found")
        with cols[2]:
            if stitch_crop is not None: st.image(stitch_crop, caption=f"Extracted Stitching (Dist: {s_stats[0]:.2f})")
            else: st.warning("Stitching not found")
            
        st.markdown("### Authentication Verdict")
        if prob_original >= 0.5:
            st.success(f"✅ AUTHENTIC (Confidence: {prob_original*100:.2f}%)")
        else:
            st.error(f"🚨 COUNTERFEIT (Confidence: {(1.0-prob_original)*100:.2f}%)")
            
        st.metric(label="Inference Latency", value=f"{exec_time:.2f} ms")
