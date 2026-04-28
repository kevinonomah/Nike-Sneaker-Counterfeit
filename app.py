import os
import cv2
import time
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & CACHING
# ==============================================================================
st.set_page_config(page_title="Sneaker Authenticator", page_icon="👟", layout="wide")

# Automatically resolves to the folder where this script lives.
# Works locally, on a server, and in Streamlit Cloud — no hardcoded paths needed.
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
YOLO_SHOE_PATH  = os.path.join(BASE_DIR, "shoe_best.pt")
YOLO_MICRO_PATH = os.path.join(BASE_DIR, "micro_best.pt")
TOWER_PATH      = os.path.join(BASE_DIR, "siamese_tower.keras")
MLP_PATH        = os.path.join(BASE_DIR, "fusion_mlp.keras")
BANK_PATH       = os.path.join(BASE_DIR, "reference_bank.pkl")


@st.cache_resource
def load_models():
    """Loads all 4 models into memory once to prevent lag on every button click."""
    yolo_shoe  = YOLO(YOLO_SHOE_PATH)
    yolo_micro = YOLO(YOLO_MICRO_PATH)
    tower      = tf.keras.models.load_model(TOWER_PATH, compile=False)
    mlp        = tf.keras.models.load_model(MLP_PATH,   compile=False)
    return yolo_shoe, yolo_micro, tower, mlp


@st.cache_data
def load_reference_bank():
    with open(BANK_PATH, "rb") as f:
        return pickle.load(f)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def crop_box(img, box):
    """Safely crop a bounding box from an image. Returns None on invalid input."""
    if box is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def resize_norm(img, size=(128, 128)):
    """Resize and normalize a crop to float32 in [0, 1]. Returns zeros if crop is invalid."""
    if img is None or img.size == 0:
        return np.zeros(size + (3,), dtype=np.float32)
    return cv2.resize(img, size).astype(np.float32) / 255.0


def compute_stats_fast(emb, bank_embs):
    """Compute min and mean cosine distances between an embedding and a reference bank."""
    if bank_embs is None or len(bank_embs) == 0:
        return [1.0, 1.0]
    sims  = np.dot(bank_embs, emb)
    dists = 1.0 - sims
    return [float(np.min(dists)), float(np.mean(dists))]


# ==============================================================================
# CORE PIPELINE — PER-IMAGE FUNCTION
# ==============================================================================
def authenticate_single(img_array, yolo_shoe, yolo_micro, tower, mlp, ref_bank):
    """
    Runs the full detection + embedding + fusion pipeline on a single image.

    Returns a result dict on success, or None if no shoe was detected.
    Returning None instead of raising allows the caller to silently skip that
    angle during ensemble aggregation without polluting the score pool.
    """
    # ── Stage 1: Whole-shoe detection ─────────────────────────────────────
    res_shoe = yolo_shoe(img_array, conf=0.25, verbose=False)[0]

    if len(res_shoe.boxes) == 0:
        return None  # No shoe found — skip this angle in the ensemble

    best_idx  = int(np.argmax(res_shoe.boxes.conf.cpu().numpy()))
    shoe_box  = res_shoe.boxes.xyxy.cpu().numpy()[best_idx]
    shoe_conf = float(res_shoe.boxes.conf.cpu().numpy()[best_idx])
    shoe_cls  = int(res_shoe.boxes.cls.cpu().numpy()[best_idx])
    shoe_crop = crop_box(img_array, shoe_box)

    # ── Stage 2: Micro-feature detection (logo + stitching) ───────────────
    res_micro = yolo_micro(shoe_crop, verbose=False)[0]
    logo_box, stitch_box = None, None
    max_l, max_s         = -1, -1

    if len(res_micro.boxes) > 0:
        confs = res_micro.boxes.conf.cpu().numpy()
        boxes = res_micro.boxes.xyxy.cpu().numpy()
        clss  = res_micro.boxes.cls.cpu().numpy().astype(int)
        for i, c in enumerate(clss):
            if c == 0 and confs[i] > max_l:
                logo_box, max_l = boxes[i], confs[i]
            elif c == 1 and confs[i] > max_s:
                stitch_box, max_s = boxes[i], confs[i]

    logo_crop   = crop_box(shoe_crop, logo_box)
    stitch_crop = crop_box(shoe_crop, stitch_box)

    # ── Stage 3: Siamese embeddings ───────────────────────────────────────
    logo_img   = resize_norm(logo_crop)
    stitch_img = resize_norm(stitch_crop)

    batch_imgs           = np.stack([logo_img, stitch_img])
    embeddings           = tower(batch_imgs, training=False).numpy()
    logo_emb, stitch_emb = embeddings[0], embeddings[1]

    # ── Stage 4: Distance stats vs. reference bank ────────────────────────
    l_stats = compute_stats_fast(logo_emb,   ref_bank.get("ShoeLogo", []))
    s_stats = compute_stats_fast(stitch_emb, ref_bank.get("StitchingPatterns", []))

    # ── Stage 5: Feature fusion vector ────────────────────────────────────
    shoe_h, shoe_w = shoe_crop.shape[:2]
    img_h,  img_w  = img_array.shape[:2]
    pred_one_hot   = np.zeros(4, dtype=np.float32)
    if 0 <= shoe_cls < 4:
        pred_one_hot[shoe_cls] = 1.0

    fvec = np.concatenate([
        np.array(l_stats + s_stats, dtype=np.float32),
        np.array(
            [shoe_conf,
             (shoe_h * shoe_w) / max(img_h * img_w, 1),
             shoe_w / max(shoe_h, 1)],
            dtype=np.float32
        ),
        pred_one_hot,
        logo_emb,
        stitch_emb,
    ])

    # ── Stage 6: MLP classification ───────────────────────────────────────
    prob_original = float(
        mlp(np.expand_dims(fvec, axis=0), training=False).numpy()[0][0]
    )

    return {
        "shoe_crop":   shoe_crop,
        "logo_crop":   logo_crop,
        "stitch_crop": stitch_crop,
        "l_stats":     l_stats,
        "s_stats":     s_stats,
        "prob":        prob_original,
    }


# ==============================================================================
# UI LAYOUT & LOGIC
# ==============================================================================
st.title("👟 Deep CNN-Siamese Sneaker Authenticator")
st.markdown(
    "**Master's Thesis Prototype** | "
    "*Upload multiple angles of a Nike Air Force 1 or Jordan 1 for ensemble analysis.*"
)

# Load assets silently in the background
with st.spinner("Loading Deep Learning Models…"):
    yolo_shoe, yolo_micro, tower, mlp = load_models()
    ref_bank = load_reference_bank()

# ── Multi-file uploader ────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload Sneaker Images (multiple angles recommended — e.g. side, heel, toe box, sole)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    # ── Preview grid ──────────────────────────────────────────────────────
    st.markdown("#### Uploaded Angles")
    preview_cols = st.columns(len(uploaded_files))
    for col, uf in zip(preview_cols, uploaded_files):
        col.image(uf, use_container_width=True, caption=uf.name)

    # ── Authentication button ─────────────────────────────────────────────
    if st.button("Authenticate Sneaker", type="primary"):
        start_time  = time.perf_counter()
        all_results = []   # collects per-angle result dicts
        skipped     = []   # filenames where no shoe was detected

        for uf in uploaded_files:
            image     = Image.open(uf).convert("RGB")
            img_array = np.array(image)

            with st.spinner(f"Analyzing {uf.name}…"):
                result = authenticate_single(
                    img_array,
                    yolo_shoe, yolo_micro,
                    tower, mlp,
                    ref_bank,
                )

            if result is None:
                skipped.append(uf.name)
            else:
                result["filename"] = uf.name
                all_results.append(result)

        exec_time = (time.perf_counter() - start_time) * 1000

        # ── Guard: no usable angles ───────────────────────────────────────
        if not all_results:
            st.error(
                "No shoe was detected in any of the uploaded images. "
                "Please try clearer or closer angles."
            )
            st.stop()

        if skipped:
            st.warning(
                f"⚠️ No shoe detected in: **{', '.join(skipped)}** "
                f"— excluded from ensemble."
            )

        # ── Per-angle breakdown ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("Per-Angle Breakdown")

        for r in all_results:
            st.markdown(
                f"**{r['filename']}** "
                f"— individual score: `{r['prob'] * 100:.1f}%`"
            )
            cols = st.columns(3)
            cols[0].image(r["shoe_crop"],   caption="Detected Shoe",   use_container_width=True)

            if r["logo_crop"] is not None:
                cols[1].image(
                    r["logo_crop"],
                    caption=f"Logo (min dist: {r['l_stats'][0]:.3f})",
                    use_container_width=True,
                )
            else:
                cols[1].warning("Logo not detected")

            if r["stitch_crop"] is not None:
                cols[2].image(
                    r["stitch_crop"],
                    caption=f"Stitching (min dist: {r['s_stats'][0]:.3f})",
                    use_container_width=True,
                )
            else:
                cols[2].warning("Stitching not detected")

        # ── Ensemble verdict (mean-pool across valid angles) ──────────────
        probs         = [r["prob"] for r in all_results]
        ensemble_prob = float(np.mean(probs))

        st.markdown("---")
        st.subheader("Ensemble Authentication Verdict")
        st.caption(
            f"Mean-pooled over **{len(all_results)}** valid angle(s) "
            f"| {len(skipped)} skipped"
        )

        if ensemble_prob >= 0.5:
            st.success(
                f"✅ AUTHENTIC  —  Ensemble Confidence: {ensemble_prob * 100:.2f}%"
            )
        else:
            st.error(
                f"🚨 COUNTERFEIT  —  Ensemble Confidence: {(1.0 - ensemble_prob) * 100:.2f}%"
            )

        # ── Per-angle score table ─────────────────────────────────────────
        st.markdown("#### Per-Angle Score Summary")
        score_data = {
            "Angle":                    [r["filename"] for r in all_results],
            "Authenticity Score (%)":   [f"{r['prob'] * 100:.2f}" for r in all_results],
            "Logo Min Dist":            [f"{r['l_stats'][0]:.4f}" for r in all_results],
            "Stitch Min Dist":          [f"{r['s_stats'][0]:.4f}" for r in all_results],
        }
        st.table(score_data)

        # ── Latency metrics ───────────────────────────────────────────────
        col_m1, col_m2 = st.columns(2)
        col_m1.metric(label="Total Inference Latency", value=f"{exec_time:.2f} ms")
        col_m2.metric(label="Angles Processed",        value=len(all_results))
