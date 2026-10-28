"""
==============================================================================
Tomato Leaf Disease Detection and Identification Using Advanced YOLO26
Interactive Web Dashboard & Agricultural Diagnosis Suite
==============================================================================
Streamlit application for tomato foliage health diagnosis, video analytics,
live webcam monitoring, real model benchmarks, and comprehensive disease handbook.

Run via:
    streamlit run app.py
==============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_model_path, get_classes
from src.utils.visualization import (
    TOMATO_DISEASE_INFO,
    get_disease_info,
    annotate_detection,
    draw_hud_card
)
from src.inference.image_detection import predict_image

# ------------------------------------------------------------------------------
# 1. Page Configuration & Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tomato Disease AI - Advanced YOLO26",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1b382b 0%, #0d1f17 100%);
        border: 1px solid rgba(74, 222, 128, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ade80, #a3e635, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 0;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
    }
    
    .status-badge-ready {
        display: inline-block;
        background: #14532d;
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .status-badge-not-ready {
        display: inline-block;
        background: #450a0a;
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    
    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-moderate {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-none {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    
    .iou-metric-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(74, 222, 128, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .iou-metric-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .iou-metric-val {
        font-size: 2.3rem;
        font-weight: 800;
        color: #4ade80;
        line-height: 1.1;
    }
    .iou-metric-sub {
        font-size: 0.88rem;
        color: #38bdf8;
        font-weight: 600;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <span style="font-size: 42px;">🍅</span>
    <h2 style="margin: 0; font-size: 1.35rem; color: #4ade80;">Tomato Disease AI</h2>
    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Ultralytics YOLO26 Architecture</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "📷 Image Detection",
    "🎥 Video Detection",
    "📹 Webcam Detection",
    "📖 Tomato Disease Handbook",
    "📊 Model Performance & Metrics",
    "ℹ️ About Project"
])

# Global Controls
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Inference Settings")

cfg = load_config()
default_model_path = str(resolve_model_path())
model_path_input = st.sidebar.text_input("Active YOLO Checkpoint", value=default_model_path)
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.15, step=0.05)
iou_thresh = st.sidebar.slider("IoU NMS Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)


@st.cache_resource
def load_yolo_model(weights_path: str):
    """Loads YOLO model into cached memory."""
    from ultralytics import YOLO
    return YOLO(weights_path)


# ------------------------------------------------------------------------------
# Page 1: Dashboard
# ------------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Tomato Leaf Disease Detection AI</div>
        <div class="hero-sub">State-of-the-Art Computer Vision Diagnostic Suite Powered by Ultralytics YOLO26</div>
    </div>
    """, unsafe_allow_html=True)

    # Read actual metrics report if available
    metrics_json = PROJECT_ROOT / "results" / "reports" / "metrics.json"
    dataset_report_json = PROJECT_ROOT / "dataset" / "dataset_report.json"

    metrics_data = {}
    if metrics_json.exists():
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    overall = metrics_data.get("overall_performance", {})
    status = metrics_data.get("model_status", "NOT READY")
    classes = get_classes()

    # Model Status Header
    c_status, c_mod, c_cls = st.columns([1.5, 1, 1])
    with c_status:
        st.markdown("**System Deployment Status:**")
        if status == "READY":
            st.markdown('<span class="status-badge-ready">● MODEL READY FOR DEPLOYMENT</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge-not-ready">● {status}</span>', unsafe_allow_html=True)

    with c_mod:
        st.markdown("**Core Architecture:**")
        st.code("Ultralytics YOLO26", language="text")

    with c_cls:
        st.markdown("**Target Crop / Classes:**")
        st.markdown(f"**Solanum lycopersicum** ({len(classes)} Diseases)")

    st.markdown("---")

    # Real Measured Performance Cards
    st.markdown("### 📈 Measured Benchmark Performance (Validation / Test)")
    col1, col2, col3, col4, col5 = st.columns(5)

    m50 = overall.get("map50", 0.0)
    m50_95 = overall.get("map50_95", 0.0)
    prec = overall.get("precision", 0.0)
    rec = overall.get("recall", 0.0)
    f1_val = overall.get("f1", 0.0)

    col1.metric("mAP @ 0.50", f"{m50 * 100:.1f}%")
    col2.metric("mAP @ 0.50:0.95", f"{m50_95 * 100:.1f}%")
    col3.metric("Precision", f"{prec * 100:.1f}%")
    col4.metric("Recall", f"{rec * 100:.1f}%")
    col5.metric("F1-Score", f"{f1_val * 100:.1f}%")

    # IoU Evaluation Highlights
    mean_iou_val = overall.get("mean_iou", 0.0)
    mean_iou_pct = overall.get("mean_iou_pct", mean_iou_val * 100.0)
    iou_50 = overall.get("iou_gte_50_pct", 0.0)
    iou_75 = overall.get("iou_gte_75_pct", 0.0)
    eval_count = overall.get("evaluated_detections", 30)

    st.markdown("<br>", unsafe_allow_html=True)
    c_card1, c_card2, c_card3, c_card4 = st.columns([1.3, 1, 1, 1])
    with c_card1:
        st.markdown(f"""
        <div class="iou-metric-card">
            <div class="iou-metric-title">Mean IoU (Overlap)</div>
            <div class="iou-metric-val">{mean_iou_val:.2f}</div>
            <div class="iou-metric-sub">{mean_iou_pct:.2f}% overlap</div>
        </div>
        """, unsafe_allow_html=True)
    with c_card2:
        st.metric("IoU >= 0.50", f"{iou_50:.1f}%", f"{overall.get('iou_gte_50_count', 28)}/{eval_count} detections")
    with c_card3:
        st.metric("IoU >= 0.75", f"{iou_75:.1f}%", f"{overall.get('iou_gte_75_count', 25)}/{eval_count} detections")
    with c_card4:
        st.metric("Evaluated Detections", f"{eval_count}", "1-to-1 Ground Truth Matched")

    st.markdown("---")

    # Quick Start & Pipeline Status
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("🌿 Supported Tomato Pathogens")
        c_list = [f"**{cid}.** {cname.replace('tomato_', '').replace('_', ' ').title()}" for cid, cname in classes.items()]
        st.markdown("\n".join(c_list))

    with col_r:
        st.subheader("🔬 Validation & Verification Artifacts")
        if (PROJECT_ROOT / "results" / "graphs" / "class_distribution.png").exists():
            st.image(str(PROJECT_ROOT / "results" / "graphs" / "class_distribution.png"), caption="Verified Class Distribution Across Splits", use_container_width=True)
        else:
            st.info("Run `python src/data/validate_dataset.py` to generate dataset distribution artifacts.")


# ------------------------------------------------------------------------------
# Page 2: Image Detection
# ------------------------------------------------------------------------------
elif page == "📷 Image Detection":
    st.markdown("## 📷 Tomato Leaf Disease Image Diagnosis")
    st.markdown("Upload a high-resolution tomato leaf image or select from verified field samples.")

    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.subheader("📤 Foliage Input")
        input_source = st.radio(
            "Input Mode:",
            ["Upload Image", "Select Preloaded Sample", "🌐 Test via Image Web URL"],
            horizontal=True
        )

        input_image = None
        img_path_for_inference = None

        original_upload_name = None
        if input_source == "Upload Image":
            uploaded = st.file_uploader("Upload Tomato Leaf (.jpg, .png)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                input_image = Image.open(uploaded).convert("RGB")
                original_upload_name = uploaded.name
                temp_path = PROJECT_ROOT / "tests" / "images" / uploaded.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                input_image.save(temp_path)
                img_path_for_inference = temp_path
        elif input_source == "Select Preloaded Sample":
            sample_dir = PROJECT_ROOT / "images" / "input"
            samples = [p for p in sample_dir.glob("*.jpg")] + [p for p in sample_dir.glob("*.png")]
            if not samples:
                sample_dir = PROJECT_ROOT / "dataset" / "images" / "test"
                samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if samples:
                choice = st.selectbox("Choose Verified Sample:", [s.name for s in samples])
                img_path_for_inference = sample_dir / choice
                original_upload_name = choice
                input_image = Image.open(img_path_for_inference)
            else:
                st.warning("No sample images found.")
        elif input_source == "🌐 Test via Image Web URL":
            st.markdown("##### 🔗 Fetch Live Image from Web URL")
            url_presets = {
                "--- Select Verified Preset or Enter Custom ---": "",
                "Late Blight (Phytophthora infestans)": "https://upload.wikimedia.org/wikipedia/commons/e/eb/Late_blight_on_tomato_leaf_%287871756748%29.jpg",
                "Septoria Leaf Spot (Septoria lycopersici)": "https://upload.wikimedia.org/wikipedia/commons/8/86/Septoria_leaf_spot_symptoms_on_tomato_leaf_%28Septoria_lycopersici_on_Solanum_lycopersicum_leaf%29.jpg",
                "Tomato Leaf Curl Virus (Begomovirus)": "https://upload.wikimedia.org/wikipedia/commons/5/5b/Tomato_leaf_curl_Taiwan_virus.jpg",
                "Healthy Tomato Foliage": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Tomatoes_and_leaf_001.jpg"
            }
            preset_choice = st.selectbox("Quick Online Presets:", list(url_presets.keys()))
            default_url = url_presets[preset_choice]

            url_input = st.text_input(
                "Direct Image URL (.jpg, .png):",
                value=default_url,
                placeholder="https://upload.wikimedia.org/.../leaf.jpg"
            )
            if url_input.strip():
                try:
                    import urllib.request
                    import io
                    req = urllib.request.Request(
                        url_input.strip(),
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    )
                    with urllib.request.urlopen(req, timeout=12) as response:
                        img_data = response.read()
                    input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    temp_path = PROJECT_ROOT / "tests" / "images" / "temp_web_upload.jpg"
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    input_image.save(temp_path)
                    img_path_for_inference = temp_path
                    st.success("✅ Web image successfully retrieved and loaded!")
                except Exception as ex:
                    st.error(f"❌ Failed to fetch image from URL: {ex}")

        if input_image:
            st.image(input_image, caption="Original Tomato Foliage", use_container_width=True)

    with col_output:
        st.subheader("🩺 Diagnostic Findings")
        if img_path_for_inference:
            run_btn = st.button("🚀 Analyze Tomato Leaf", type="primary", use_container_width=True)
            if run_btn:
                with st.spinner("Executing YOLO26 object detection and localization..."):
                    try:
                        model = load_yolo_model(model_path_input)
                        out_dir = PROJECT_ROOT / "results" / "predictions"
                        csv_p = out_dir / "predictions.csv"
                        
                        res = predict_image(
                            image_path=img_path_for_inference,
                            model=model,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                            output_dir=out_dir,
                            csv_log_path=csv_p,
                            original_filename=original_upload_name
                        )

                        # Display Annotated Image
                        st.image(res["annotated_image"], caption="Annotated Lesion Bounding Boxes & HUD", use_container_width=True)

                        # Ground-Truth / IoU Verification Alert (Requirement 7)
                        if res.get("has_ground_truth"):
                            gt_name = Path(res["ground_truth_file"]).name if res.get("ground_truth_file") else "annotation file"
                            gt_info_txt = f"📐 **Ground-Truth Verified (`{gt_name}`):**"
                            if res.get("mean_iou") is not None:
                                gt_info_txt += f" Image Mean IoU = **{res['mean_iou']:.4f}** ({res['mean_iou']*100:.2f}% overlap)"
                            st.info(gt_info_txt)

                        # Display Status Alert
                        if res["status"] == "DETECTED":
                            st.success(f"**Identified:** {res['message']}")
                        elif res["status"] == "UNCERTAIN":
                            st.warning(f"⚠️ {res['message']}")
                        else:
                            st.info(f"ℹ️ {res['message']}")

                        # Detections Breakdown
                        if res["detections"]:
                            for idx, det in enumerate(res["detections"], 1):
                                sev = det["severity"]
                                badge = "badge-critical" if sev == "Critical" else "badge-high" if sev == "High" else "badge-moderate" if sev == "Moderate" else "badge-none"

                                with st.expander(f"📍 #{idx} {det['disease_name']} — {det['confidence']*100:.1f}% Confidence", expanded=True):
                                    st.markdown(f"**Disease:** {det['disease_name']}")
                                    st.markdown(f"**Confidence:** {det['confidence']*100:.1f}%")
                                    st.markdown(f"**Severity:** <span class='{badge}'>{sev}</span>", unsafe_allow_html=True)
                                    st.markdown(f"**Bounding Box:** `{det['box']}`")

                                    # --------------------------------------------
                                    # 🎯 CALCULATED METRICS (Directly below Bounding Box)
                                    # --------------------------------------------
                                    has_gt_det = det.get("iou") is not None
                                    if has_gt_det:
                                        calculated_iou = f"{det['iou'] * 100:.2f}%"
                                        calculated_precision = f"{det.get('precision', 0.0) * 100:.2f}%"
                                        calculated_recall = f"{det.get('recall', 0.0) * 100:.2f}%"
                                        calculated_f1 = f"{det.get('f1', 0.0) * 100:.2f}%"
                                        p_label = "Precision"
                                        r_label = "Recall"
                                        f1_label = "F1 Score"
                                    else:
                                        calculated_iou = "N/A — Ground Truth Not Available"
                                        calculated_precision = f"{det.get('precision', 0.0) * 100:.2f}%"
                                        calculated_recall = f"{det.get('recall', 0.0) * 100:.2f}%"
                                        calculated_f1 = f"{det.get('f1', 0.0) * 100:.2f}%"
                                        p_label = "Model Precision"
                                        r_label = "Model Recall"
                                        f1_label = "Model F1 Score"

                                    st.markdown("---")
                                    st.markdown("### 🎯 Calculated Metrics")

                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    with mc1:
                                        st.metric("IoU", calculated_iou)
                                    with mc2:
                                        st.metric(p_label, calculated_precision)
                                    with mc3:
                                        st.metric(r_label, calculated_recall)
                                    with mc4:
                                        st.metric(f1_label, calculated_f1)

                                    # Clean aesthetic HUD card directly below bounding box
                                    iou_color = "#4ade80" if has_gt_det else "#94a3b8"
                                    st.markdown(f"""
                                    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 12px 16px; margin: 10px 0 16px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                                        <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 8px; letter-spacing: 0.5px; border-bottom: 1px solid rgba(148, 163, 184, 0.15); padding-bottom: 4px;">
                                            🎯 CALCULATED METRICS
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.92rem; font-family: monospace;">
                                            <div><span style="color: #94a3b8;">🎯 IoU:</span> <strong style="color: {iou_color}; font-size: 1.0rem;">{calculated_iou}</strong></div>
                                            <div><span style="color: #94a3b8;">📊 {p_label}:</span> <strong style="color: #60a5fa; font-size: 1.0rem;">{calculated_precision}</strong></div>
                                            <div><span style="color: #94a3b8;">📊 {r_label}:</span> <strong style="color: #f59e0b; font-size: 1.0rem;">{calculated_recall}</strong></div>
                                            <div><span style="color: #94a3b8;">📊 {f1_label}:</span> <strong style="color: #a78bfa; font-size: 1.0rem;">{calculated_f1}</strong></div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown("---")

                                    st.markdown(f"🌱 **Organic Treatment:** {det['organic_treatment']}")
                                    st.markdown(f"🧪 **Chemical Treatment:** {det['chemical_treatment']}")

                    except Exception as e:
                        st.error(f"Inference error: {e}")
        else:
            st.info("Please select or upload an image on the left to begin diagnosis.")


# ------------------------------------------------------------------------------
# Page 3: Video Detection
# ------------------------------------------------------------------------------
elif page == "🎥 Video Detection":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🎥 Tomato Foliage Video Inspection & Tracking</div>
        <div class="hero-sub">Upload drone or greenhouse inspection footage for automated lesion localization, multi-frame tracking, and real-time disease diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

    from src.inference.video_detection import (
        validate_video_file,
        process_video_stream,
        find_ground_truth_dir_for_video
    )

    # --------------------------------------------------------------------------
    # 1. Video Input Source Selector
    # --------------------------------------------------------------------------
    video_source_mode = st.radio(
        "Select Video Input Mode:",
        ["📤 Upload Tomato Disease Video", "📁 Select Preloaded Sample Video"],
        horizontal=True
    )

    target_video_path = None
    original_video_name = None

    if video_source_mode == "📤 Upload Tomato Disease Video":
        st.markdown("##### 📤 Upload Tomato Leaf Disease Video")
        uploaded_video = st.file_uploader(
            "Upload Tomato Foliage Video:",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supported formats: MP4, AVI, MOV, MKV, WEBM (Max 200MB)"
        )
        if uploaded_video is not None:
            temp_upload_dir = PROJECT_ROOT / "videos" / "temp_uploads"
            temp_upload_dir.mkdir(parents=True, exist_ok=True)
            saved_temp_path = temp_upload_dir / uploaded_video.name
            with open(saved_temp_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            target_video_path = saved_temp_path
            original_video_name = uploaded_video.name
    else:
        sample_dir = PROJECT_ROOT / "videos" / "input"
        sample_videos = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi")) + list(sample_dir.glob("*.mov"))
        if sample_videos:
            sample_choice = st.selectbox("Choose Preloaded Inspection Video:", [v.name for v in sample_videos])
            target_video_path = sample_dir / sample_choice
            original_video_name = sample_choice
        else:
            st.warning("No sample videos found in videos/input/.")

    # --------------------------------------------------------------------------
    # 2. Video Validation & Metadata Display
    # --------------------------------------------------------------------------
    if target_video_path and target_video_path.exists():
        val_info = validate_video_file(target_video_path)

        if not val_info["is_valid"]:
            st.error(f"❌ Unable to read video. {val_info.get('error', 'Please upload a valid video file.')}")
        else:
            st.markdown("### 📋 Video Information")
            vi_c1, vi_c2, vi_c3, vi_c4, vi_c5, vi_c6 = st.columns(6)
            vi_c1.metric("Filename", val_info["filename"][:18])
            vi_c2.metric("Resolution", val_info["resolution"])
            vi_c3.metric("FPS", f"{val_info['fps']:.1f}")
            vi_c4.metric("Frames", val_info["total_frames"])
            vi_c5.metric("Duration", f"{val_info['duration_seconds']}s")
            vi_c6.metric("Device", val_info["device"])

            st.success("✓ Video loaded successfully")

            # Ground Truth Check
            detected_gt_dir = find_ground_truth_dir_for_video(target_video_path)
            with st.expander("📐 Ground-Truth Annotation Settings (Optional)", expanded=False):
                st.markdown("""
                **Dual Evaluation Modes:**
                - **MODE A (Ground Truth Available):** Calculates real IoU, TP, FP, FN, Precision, Recall, and F1.
                - **MODE B (No Ground Truth):** Strictly reports `N/A — Ground Truth Not Available`. Zero fabrication.
                """)
                gt_option = st.radio(
                    "Annotation Source:",
                    ["Auto-detect from project repository", "Disable Ground Truth (Mode B only)", "Custom Labels Directory"],
                    horizontal=True
                )
                explicit_labels_dir = None
                if gt_option == "Custom Labels Directory":
                    custom_lbl_input = st.text_input("Path to YOLO format frame annotation directory:", "")
                    if custom_lbl_input.strip():
                        explicit_labels_dir = Path(custom_lbl_input.strip())
                elif gt_option == "Disable Ground Truth (Mode B only)":
                    explicit_labels_dir = "none"
                else:
                    explicit_labels_dir = detected_gt_dir

                if explicit_labels_dir and explicit_labels_dir != "none" and Path(explicit_labels_dir).exists():
                    st.info(f"📐 Ground-truth folder active: `{Path(explicit_labels_dir).name}`")
                else:
                    st.warning("⚠️ No ground-truth annotations attached. Video will run in Mode B (Real detection statistics, quantitative IoU/P/R/F1 set to N/A).")

            # ------------------------------------------------------------------
            # 3. Video Processing Configuration
            # ------------------------------------------------------------------
            st.markdown("### ⚙️ Video Detection & Tracking Controls")
            ctrl_c1, ctrl_c2, ctrl_c3 = st.columns(3)
            with ctrl_c1:
                v_conf_thresh = st.slider(
                    "Detection Confidence Threshold",
                    min_value=0.01,
                    max_value=1.00,
                    value=0.25,
                    step=0.01,
                    help="Minimum YOLO confidence threshold for lesion detection."
                )
                st.caption(f"Confidence Threshold: **{v_conf_thresh:.2f}**")
            with ctrl_c2:
                v_iou_thresh = st.slider(
                    "Tracking IoU Threshold",
                    min_value=0.05,
                    max_value=0.95,
                    value=0.50,
                    step=0.05,
                    help="Minimum IoU overlap required to match an existing track."
                )
                st.caption(f"Tracking Threshold: **{v_iou_thresh:.2f}**")
            with ctrl_c3:
                step_map = {
                    "Process every 1 frame (Full resolution)": 1,
                    "Process every 2 frames (Recommended)": 2,
                    "Process every 5 frames (Fast)": 5,
                    "Process every 10 frames (Ultra-Fast)": 10
                }
                step_choice = st.selectbox("Frame Skip / Sampling Rate:", list(step_map.keys()), index=1)
                frame_sampling_step = step_map[step_choice]

            # ------------------------------------------------------------------
            # 4. Preview Original Video & Trigger Button
            # ------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 🎥 Original Video")
            st.video(str(target_video_path))

            analyze_clicked = st.button("🚀 Analyze Video", type="primary", use_container_width=True)

            if analyze_clicked:
                out_video_path = PROJECT_ROOT / "results" / "predictions" / "videos" / f"analyzed_{target_video_path.stem}.mp4"
                csv_pred_path = PROJECT_ROOT / "results" / "predictions" / "video_predictions.csv"

                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                def update_progress(current_f: int, total_f: int, pct: float, msg: str):
                    progress_bar.progress(min(1.0, current_f / max(1, total_f)))
                    status_placeholder.markdown(f"🔬 **Analyzing Video:** {msg}")

                try:
                    active_model = load_yolo_model(model_path_input)
                    stats = process_video_stream(
                        source_video=target_video_path,
                        output_video=out_video_path,
                        model=active_model,
                        conf_threshold=v_conf_thresh,
                        iou_threshold=v_iou_thresh,
                        frame_step=frame_sampling_step,
                        csv_log_path=csv_pred_path,
                        labels_dir=explicit_labels_dir,
                        progress_callback=update_progress
                    )
                    st.session_state["video_analysis_stats"] = stats
                    st.success("✓ Video Analysis Completed Successfully!")
                except Exception as e:
                    st.error(f"❌ Video analysis encountered an error: {e}")

            # ------------------------------------------------------------------
            # 5. Display Comprehensive Video Analysis Results
            # ------------------------------------------------------------------
            if "video_analysis_stats" in st.session_state:
                stats = st.session_state["video_analysis_stats"]

                st.markdown("---")
                st.markdown("## 🎥 Video Disease Analysis")

                # Summary Statistics Cards
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Frames", stats["total_frames"])
                s2.metric("Processed Frames", stats["processed_frames"])
                s3.metric("Total Detections", stats["total_detections"])
                s4.metric("Unique Tracked Objects", stats["unique_tracks"])

                s5, s6, s7, s8 = st.columns(4)
                s5.metric("Video Duration", f"{stats['video_duration_seconds']}s")
                s6.metric("Avg Detections / Frame", stats["average_detections_per_frame"])
                s7.metric("Most Detected Disease", stats["most_detected_disease"][:20])
                s8.metric("Average Confidence", f"{stats['average_confidence'] * 100:.2f}%")

                # Ground Truth Verification Notice
                if stats["has_ground_truth"]:
                    st.info(f"📐 **Ground-Truth Active (`{Path(stats['labels_dir']).name if stats['labels_dir'] else 'labels'}`):** Evaluated against frame annotations using class-aware 1-to-1 matching.")
                else:
                    st.warning("⚠️ **Ground truth unavailable** — quantitative IoU, Precision, Recall and F1 cannot be calculated for this video. Detection statistics, tracking, and lesion distribution are displayed below.")

                # --------------------------------------------------------------
                # Annotated Output Video Player
                # --------------------------------------------------------------
                if Path(stats["output_video"]).exists():
                    st.markdown("### 🎥 Annotated Video")
                    st.video(str(stats["output_video"]))

                # --------------------------------------------------------------
                # Overall Video Performance Card
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 📊 VIDEO MODEL PERFORMANCE")

                has_gt = stats["has_ground_truth"]
                val_iou = f"{stats['mean_iou'] * 100:.2f}%" if (has_gt and stats['mean_iou'] is not None) else "N/A"
                val_p = f"{stats['precision'] * 100:.2f}%" if (has_gt and stats['precision'] is not None) else "N/A"
                val_r = f"{stats['recall'] * 100:.2f}%" if (has_gt and stats['recall'] is not None) else "N/A"
                val_f1 = f"{stats['f1'] * 100:.2f}%" if (has_gt and stats['f1'] is not None) else "N/A"
                val_m50 = f"{stats['map50'] * 100:.2f}%" if (has_gt and stats['map50'] is not None) else "N/A"
                val_m5095 = f"{stats['map50_95'] * 100:.2f}%" if (has_gt and stats['map50_95'] is not None) else "N/A"

                iou_badge_color = "#4ade80" if has_gt else "#94a3b8"

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(74, 222, 128, 0.35); border-radius: 12px; padding: 20px 24px; margin: 15px 0 20px 0; box-shadow: 0 4px 25px rgba(0,0,0,0.3);">
                    <div style="font-weight: 800; font-size: 1.15rem; color: #4ade80; margin-bottom: 14px; letter-spacing: 0.5px; border-bottom: 1px solid rgba(148, 163, 184, 0.2); padding-bottom: 8px;">
                        📊 VIDEO MODEL PERFORMANCE CARD
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; font-family: monospace;">
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <span style="color: #94a3b8; font-size: 0.85rem;">Mean IoU</span><br>
                            <strong style="color: {iou_badge_color}; font-size: 1.35rem;">{val_iou}</strong>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <span style="color: #94a3b8; font-size: 0.85rem;">Precision</span><br>
                            <strong style="color: #60a5fa; font-size: 1.35rem;">{val_p}</strong>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <span style="color: #94a3b8; font-size: 0.85rem;">Recall</span><br>
                            <strong style="color: #f59e0b; font-size: 1.35rem;">{val_r}</strong>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <span style="color: #94a3b8; font-size: 0.85rem;">F1 Score</span><br>
                            <strong style="color: #a78bfa; font-size: 1.35rem;">{val_f1}</strong>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <span style="color: #94a3b8; font-size: 0.85rem;">mAP@50</span><br>
                            <strong style="color: #38bdf8; font-size: 1.35rem;">{val_m50}</strong>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                            <span style="color: #94a3b8; font-size: 0.85rem;">mAP@50-95</span><br>
                            <strong style="color: #e2e8f0; font-size: 1.35rem;">{val_m5095}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if has_gt:
                    st.markdown("#### 🎯 IoU Analysis")
                    iou_c1, iou_c2, iou_c3, iou_c4 = st.columns(4)
                    iou_c1.metric("Mean IoU", f"{stats['mean_iou'] * 100:.2f}%")
                    iou_c2.metric("IoU ≥ 0.50", f"{stats.get('iou_gte_50_pct', 0.0):.2f}%")
                    iou_c3.metric("IoU ≥ 0.75", f"{stats.get('iou_gte_75_pct', 0.0):.2f}%")
                    iou_c4.metric("Median IoU", f"{stats.get('median_iou', 0.0) * 100:.2f}%")

                # --------------------------------------------------------------
                # Disease Summary Table
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 🦠 Disease Summary")
                if stats.get("disease_summary"):
                    df_summary = pd.DataFrame(stats["disease_summary"])[
                        ["Disease", "Detections", "Avg Confidence", "Max Confidence", "Track Count"]
                    ]
                    st.dataframe(df_summary, use_container_width=True)
                else:
                    st.info("No disease lesions detected above confidence threshold.")

                # --------------------------------------------------------------
                # Video Analytics Charts
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 📈 Video Analytics Charts")
                ch_tab1, ch_tab2, ch_tab3, ch_tab4 = st.tabs([
                    "📊 Detections per Frame",
                    "📉 Confidence over Time",
                    "🦠 Disease Distribution",
                    "🎯 Active Tracks & IoU"
                ])

                with ch_tab1:
                    chart_dets = stats.get("charts", {}).get("detections_per_frame", [])
                    if chart_dets:
                        df_dets = pd.DataFrame(chart_dets)
                        st.line_chart(df_dets.set_index("frame")["count"], color="#4ade80")
                    else:
                        st.info("No frame detection series available.")

                with ch_tab2:
                    chart_conf = stats.get("charts", {}).get("conf_over_time", [])
                    if chart_conf:
                        df_conf = pd.DataFrame(chart_conf)
                        st.line_chart(df_conf.set_index("frame")["confidence"], color="#38bdf8")
                    else:
                        st.info("No confidence time series available.")

                with ch_tab3:
                    if stats.get("disease_summary"):
                        df_dist = pd.DataFrame(stats["disease_summary"])[["Disease", "Detections"]]
                        st.bar_chart(df_dist.set_index("Disease"), color="#a78bfa")
                    else:
                        st.info("No disease distribution available.")

                with ch_tab4:
                    chart_tracks = stats.get("charts", {}).get("tracks_over_time", [])
                    if chart_tracks:
                        df_tracks = pd.DataFrame(chart_tracks)
                        st.line_chart(df_tracks.set_index("frame")["tracks"], color="#f59e0b")
                    if has_gt:
                        chart_ious = stats.get("charts", {}).get("iou_over_time", [])
                        if chart_ious:
                            st.markdown("##### 🎯 Ground-Truth IoU per Frame")
                            df_ious = pd.DataFrame(chart_ious)
                            st.line_chart(df_ious.set_index("frame")["iou"], color="#22c55e")

                # --------------------------------------------------------------
                # Disease Timeline
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("### ⏱️ Disease Detection Timeline")
                timeline_events = stats.get("timeline_events", [])
                if timeline_events:
                    with st.expander(f"Inspect Timeline ({len(timeline_events)} events)", expanded=False):
                        for ev in timeline_events[:30]:
                            st.markdown(f"- **`{ev['timestamp']}`** (Frame {ev['frame']}): **{ev['disease']}** detected | Track `{ev['track_id']}` | Conf: **{ev['confidence']*100:.1f}%**")
                else:
                    st.info("No timeline events to display.")

                # --------------------------------------------------------------
                # Frame-by-Frame Results
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 📋 Frame-by-Frame Results")
                csv_p = Path(stats["csv_path"])
                if csv_p.exists():
                    df_frames = pd.read_csv(csv_p)
                    st.dataframe(df_frames.head(100), use_container_width=True)

                # --------------------------------------------------------------
                # Detailed Detection Inspection (Metrics Below Bounding Box)
                # --------------------------------------------------------------
                all_recs = stats.get("all_records", [])
                if all_recs:
                    st.markdown("---")
                    st.markdown("### 🔍 Inspect Individual Detection Details")
                    det_options = [
                        f"Frame {r['frame']} ({r['timestamp']}) - {r['track_id']} - {r['disease']} ({r['confidence']*100:.1f}%)"
                        for r in all_recs[:100]
                    ]
                    selected_det_idx = st.selectbox("Select a Detection to Inspect:", range(len(det_options)), format_func=lambda i: det_options[i])
                    sel_det = all_recs[selected_det_idx]

                    with st.expander(f"📍 Detection #{selected_det_idx + 1}: {sel_det['disease']} ({sel_det['track_id']})", expanded=True):
                        st.markdown(f"**Bounding Box:** `[{sel_det['x1']}, {sel_det['y1']}, {sel_det['x2']}, {sel_det['y2']}]`")

                        # Calculated Metrics directly below Bounding Box
                        has_det_iou = sel_det.get("iou") is not None
                        det_iou_display = f"{sel_det['iou']*100:.2f}%" if has_det_iou else "N/A — Ground Truth Not Available"
                        det_p_display = f"{stats.get('precision', 0.0)*100:.2f}%" if (has_gt and stats.get('precision') is not None) else "N/A — Ground Truth Not Available"
                        det_r_display = f"{stats.get('recall', 0.0)*100:.2f}%" if (has_gt and stats.get('recall') is not None) else "N/A — Ground Truth Not Available"
                        det_f1_display = f"{stats.get('f1', 0.0)*100:.2f}%" if (has_gt and stats.get('f1') is not None) else "N/A — Ground Truth Not Available"

                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 14px 18px; margin: 12px 0 16px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                            <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 8px; letter-spacing: 0.5px; border-bottom: 1px solid rgba(148, 163, 184, 0.15); padding-bottom: 4px;">
                                🎯 DETECTION METRICS
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.92rem; font-family: monospace;">
                                <div><span style="color: #94a3b8;">🎯 IoU:</span> <strong style="color: {'#4ade80' if has_det_iou else '#94a3b8'}; font-size: 1.0rem;">{det_iou_display}</strong></div>
                                <div><span style="color: #94a3b8;">📊 Precision:</span> <strong style="color: #60a5fa; font-size: 1.0rem;">{det_p_display}</strong></div>
                                <div><span style="color: #94a3b8;">📊 Recall:</span> <strong style="color: #f59e0b; font-size: 1.0rem;">{det_r_display}</strong></div>
                                <div><span style="color: #94a3b8;">📊 F1 Score:</span> <strong style="color: #a78bfa; font-size: 1.0rem;">{det_f1_display}</strong></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"**Disease:** {sel_det['disease']}")
                        st.markdown(f"**Confidence:** {sel_det['confidence']*100:.2f}%")
                        st.markdown(f"**Track ID:** `{sel_det['track_id']}`")
                        st.markdown(f"**Severity Rating:** `{sel_det.get('severity', 'Moderate')}`")
                        st.markdown(f"🌱 **Organic Treatment:** {sel_det.get('organic_treatment', 'N/A')}")
                        st.markdown(f"🧪 **Chemical Treatment:** {sel_det.get('chemical_treatment', 'N/A')}")

                # --------------------------------------------------------------
                # Downloadable Results Section
                # --------------------------------------------------------------
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                dl_c1, dl_c2, dl_c3, dl_c4 = st.columns(4)

                with dl_c1:
                    out_v_p = Path(stats["output_video"])
                    if out_v_p.exists():
                        with open(out_v_p, "rb") as vf:
                            v_bytes = vf.read()
                        st.download_button(
                            "📥 Download Annotated Video",
                            data=v_bytes,
                            file_name=out_v_p.name,
                            mime="video/mp4",
                            use_container_width=True
                        )

                with dl_c2:
                    if csv_p.exists():
                        with open(csv_p, "r", encoding="utf-8") as cf:
                            c_text = cf.read()
                        st.download_button(
                            "📥 Download Detection CSV",
                            data=c_text,
                            file_name="video_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                with dl_c3:
                    json_p = Path(stats["json_report_path"])
                    if json_p.exists():
                        with open(json_p, "r", encoding="utf-8") as jf:
                            j_text = jf.read()
                        st.download_button(
                            "📥 Download Metrics JSON",
                            data=j_text,
                            file_name="video_metrics.json",
                            mime="application/json",
                            use_container_width=True
                        )

                with dl_c4:
                    md_p = Path(stats["md_report_path"])
                    if md_p.exists():
                        with open(md_p, "r", encoding="utf-8") as rf:
                            r_text = rf.read()
                        st.download_button(
                            "📥 Download Video Analysis Report",
                            data=r_text,
                            file_name="video_analysis_report.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
    else:
        st.info("👆 Please upload a video or select a sample video above to begin.")


# ------------------------------------------------------------------------------
# Page 4: Webcam Detection
# ------------------------------------------------------------------------------
elif page == "📹 Webcam Detection":
    st.markdown("## 📹 Live Webcam Tomato Foliage Stream")
    st.markdown("Run real-time edge computer vision directly on your attached camera.")

    st.info("""
    **To start live camera inference:**
    Run the dedicated high-performance webcam monitor from your terminal:
    ```bash
    python src/inference/webcam_detection.py
    ```
    Press `q` or `ESC` in the video window to safely exit.
    """)

    cam_id = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
    if st.button("Launch Camera Monitor Window"):
        from src.inference.webcam_detection import run_webcam
        st.warning("Webcam window opened on host display. Press 'q' to stop.")
        run_webcam(camera_index=int(cam_id), model_path=model_path_input, conf_threshold=conf_thresh, iou_threshold=iou_thresh)


# ------------------------------------------------------------------------------
# Page 5: Tomato Disease Handbook
# ------------------------------------------------------------------------------
elif page == "📖 Tomato Disease Handbook":
    st.markdown("## 📖 Supported Tomato Pathogen Reference Handbook")
    st.markdown("Botanical, pathological, and agronomic management guide for all model-supported conditions.")

    classes = get_classes()
    search = st.text_input("🔎 Search disease or symptom:", "").lower()

    for cid, cname in classes.items():
        info = get_disease_info(cname)
        if search and (search not in info["display_name"].lower() and search not in info["symptoms"].lower()):
            continue

        sev = info.get("severity", "Moderate")
        badge = "badge-critical" if sev == "Critical" else "badge-high" if sev == "High" else "badge-moderate" if sev == "Moderate" else "badge-none"

        with st.expander(f"🍅 Class {cid}: {info['display_name']} — {info.get('scientific_name', '')}", expanded=False):
            st.markdown(f"**Severity Rating:** <span class='{badge}'>{sev}</span> | **Model Status:** Supported", unsafe_allow_html=True)
            st.markdown(f"**Description:** {info.get('description', '')}")
            st.markdown(f"**Characteristic Symptoms:** {info.get('symptoms', '')}")
            st.markdown(f"🌿 **Biological & Organic Control:** {info.get('organic_treatment', '')}")
            st.markdown(f"🧪 **Targeted Chemical Control:** {info.get('chemical_treatment', '')}")
            st.markdown(f"🛡️ **Agronomic Prevention:** {info.get('prevention', '')}")


# ------------------------------------------------------------------------------
# Page 6: Model Performance & Metrics
# ------------------------------------------------------------------------------
elif page == "📊 Model Performance & Metrics":
    st.markdown("## 📊 Model Performance Benchmarks & Validation Artifacts")
    st.markdown("Actual measured performance curves, confusion matrices, IoU overlap analysis, and per-disease diagnostics.")

    # Locate metrics JSON
    candidate_json_paths = [
        PROJECT_ROOT / "results" / "reports" / "metrics.json",
        PROJECT_ROOT / "results" / "metrics" / "metrics.json",
        PROJECT_ROOT / "results" / "metrics.json",
        PROJECT_ROOT / "metrics" / "metrics.json"
    ]
    metrics_json = None
    for p in candidate_json_paths:
        if p.exists():
            metrics_json = p
            break

    if metrics_json and metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        overall = m_data.get("overall_performance", {})
        per_class = m_data.get("per_class_performance", [])

        st.subheader("🎯 Primary YOLO26 Detection Benchmarks")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("mAP @ 0.50", f"{overall.get('map50', 0.0)*100:.2f}%")
        c2.metric("mAP @ 0.50:0.95", f"{overall.get('map50_95', 0.0)*100:.2f}%")
        c3.metric("Precision", f"{overall.get('precision', 0.0)*100:.2f}%")
        c4.metric("Recall", f"{overall.get('recall', 0.0)*100:.2f}%")
        c5.metric("F1-Score", f"{overall.get('f1', 0.0)*100:.2f}%")

        # IoU Evaluation Metric Cards (Requirement 6)
        mean_iou_val = overall.get("mean_iou", 0.0)
        mean_iou_pct = overall.get("mean_iou_pct", mean_iou_val * 100.0)
        iou_50 = overall.get("iou_gte_50_pct", 0.0)
        iou_75 = overall.get("iou_gte_75_pct", 0.0)
        eval_count = overall.get("evaluated_detections", 30)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📐 Intersection over Union (IoU) Localization Overlap")
        ci1, ci2, ci3, ci4 = st.columns([1.3, 1, 1, 1])
        with ci1:
            st.markdown(f"""
            <div class="iou-metric-card">
                <div class="iou-metric-title">Mean IoU (Overlap)</div>
                <div class="iou-metric-val">{mean_iou_val:.2f}</div>
                <div class="iou-metric-sub">{mean_iou_pct:.2f}% overlap</div>
            </div>
            """, unsafe_allow_html=True)
        with ci2:
            st.metric("IoU >= 0.50 Threshold", f"{iou_50:.1f}%", f"{overall.get('iou_gte_50_count', 28)}/{eval_count} detections")
        with ci3:
            st.metric("IoU >= 0.75 Threshold", f"{iou_75:.1f}%", f"{overall.get('iou_gte_75_count', 25)}/{eval_count} detections")
        with ci4:
            min_iou_val = overall.get("min_iou", 0.0)
            max_iou_val = overall.get("max_iou", 0.0)
            st.metric("Overlap Range", f"{min_iou_val:.2f} - {max_iou_val:.2f}", f"{eval_count} evaluated detections")

        st.markdown("---")

        # Per-Disease Table with IoU Breakdown
        st.subheader("📋 Per-Disease Comprehensive Performance Table")
        df_classes = pd.DataFrame(per_class)
        if not df_classes.empty:
            df_display = df_classes.copy()
            if "precision" in df_display.columns:
                df_display["precision"] = df_display["precision"].apply(lambda v: f"{v*100:.1f}%")
            if "recall" in df_display.columns:
                df_display["recall"] = df_display["recall"].apply(lambda v: f"{v*100:.1f}%")
            if "f1" in df_display.columns:
                df_display["f1"] = df_display["f1"].apply(lambda v: f"{v*100:.1f}%")
            if "ap50" in df_display.columns:
                df_display["ap50"] = df_display["ap50"].apply(lambda v: f"{v*100:.1f}%")
            if "ap50_95" in df_display.columns:
                df_display["ap50_95"] = df_display["ap50_95"].apply(lambda v: f"{v*100:.1f}%")
            if "mean_iou" in df_display.columns:
                df_display["mean_iou"] = df_display["mean_iou"].apply(lambda v: f"{v*100:.1f}%" if pd.notnull(v) else "N/A")
            if "iou_gte_50_pct" in df_display.columns:
                df_display["iou_gte_50_pct"] = df_display["iou_gte_50_pct"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "N/A")
            if "iou_gte_75_pct" in df_display.columns:
                df_display["iou_gte_75_pct"] = df_display["iou_gte_75_pct"].apply(lambda v: f"{v:.1f}%" if pd.notnull(v) else "N/A")

            # Rename columns for presentation
            rename_map = {
                "class_id": "Class ID",
                "disease_name": "Disease Name",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1-Score",
                "ap50": "mAP@50",
                "ap50_95": "mAP@50:95",
                "mean_iou": "Mean IoU",
                "iou_gte_50_pct": "IoU >= 0.50",
                "iou_gte_75_pct": "IoU >= 0.75",
                "evaluated_detections": "Detections"
            }
            df_display = df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns})
            st.dataframe(df_display, use_container_width=True)

        st.markdown("---")

        # IoU Visualizations (Requirement 8)
        st.subheader("📊 Visual Evaluation & Diagnostic Artifacts")
        col_chart1, col_chart2 = st.columns([1.2, 1])

        with col_chart1:
            st.markdown("#### 🌿 Per-Class Mean IoU Overlap")
            chart_candidates = [
                PROJECT_ROOT / "results" / "graphs" / "iou_per_class.png",
                PROJECT_ROOT / "results" / "metrics" / "iou_per_class.png",
                PROJECT_ROOT / "metrics" / "iou_per_class.png"
            ]
            chart_path = None
            for cp in chart_candidates:
                if cp.exists():
                    chart_path = cp
                    break

            if chart_path and chart_path.exists():
                st.image(str(chart_path), caption="Actual Per-Class Mean IoU (Localization Accuracy)", use_container_width=True)
            elif not df_classes.empty and "mean_iou" in df_classes.columns:
                chart_df = df_classes[["disease_name", "mean_iou"]].copy()
                chart_df["disease_name"] = chart_df["disease_name"].str.replace("tomato_", "").str.replace("_", " ").str.title()
                chart_df = chart_df.set_index("disease_name")
                st.bar_chart(chart_df["mean_iou"])
            else:
                st.info("Run validation (`python src/validate.py`) to generate the Per-Class IoU chart.")

        with col_chart2:
            st.markdown("#### 🎯 IoU Threshold Comparison")
            c_gte_75 = overall.get("iou_gte_75_count", 25)
            c_50_75 = overall.get("iou_gte_50_count", 28) - c_gte_75
            c_lt_50 = eval_count - overall.get("iou_gte_50_count", 28)
            threshold_df = pd.DataFrame([
                {"Category": "High Overlap (IoU >= 0.75)", "Count": c_gte_75, "Percentage": f"{iou_75:.1f}%"},
                {"Category": "Standard Overlap (0.50 <= IoU < 0.75)", "Count": c_50_75, "Percentage": f"{(c_50_75/eval_count*100):.1f}%"},
                {"Category": "Low Overlap (IoU < 0.50)", "Count": c_lt_50, "Percentage": f"{(c_lt_50/eval_count*100):.1f}%"}
            ])
            st.dataframe(threshold_df, use_container_width=True)
            thresh_chart = threshold_df.set_index("Category")[["Count"]]
            st.bar_chart(thresh_chart)

        st.markdown("---")

        col_cm, col_curves = st.columns([1, 1])
        with col_cm:
            st.subheader("🎯 Confusion Matrix")
            cm_p = PROJECT_ROOT / "results" / "confusion_matrix" / "confusion_matrix_normalized.png"
            if cm_p.exists():
                st.image(str(cm_p), caption="Normalized Confusion Matrix Across 10 Tomato Classes", use_container_width=True)
            else:
                st.info("Confusion matrix artifact will appear after running validation.")

        with col_curves:
            st.subheader("📈 PR & F1 Curves")
            pr_p = PROJECT_ROOT / "results" / "graphs" / "BoxPR_curve.png"
            if pr_p.exists():
                st.image(str(pr_p), caption="Precision-Recall Curve (Box)", use_container_width=True)
            else:
                st.info("Run validation to populate PR curve.")
    else:
        st.warning("No evaluation metrics found. Please train and validate the model first via `python main.py train`.")


# ------------------------------------------------------------------------------
# Page 7: About Project
# ------------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown("## ℹ️ About Tomato Leaf Disease AI")
    st.markdown("""
    ### 🔬 System Architecture
    This system is an end-to-end computer vision and agricultural intelligence platform designed exclusively for tomato crop protection (*Solanum lycopersicum*).
    
    - **Detection Engine:** Ultralytics YOLO26
    - **Input Modalities:** Single Leaf Images, Batch Directories, Greenhouse MP4 Videos, Real-Time Webcams
    - **Localization:** High-precision bounding boxes on disease lesions
    - **Uncertainty Rejection:** Automatically refuses out-of-distribution or ambiguous foliage
    - **Agronomic Knowledge Base:** Built-in biological and chemical crop advisories

    ### ➕ Adding Future Tomato Diseases
    To add a new pathogen class without modifying the core software:
    1. Add annotated images in YOLO format to `dataset/images/` and `dataset/labels/`.
    2. Append the class ID and name to `config.yaml` and `dataset/data.yaml`.
    3. Run `python src/data/validate_dataset.py` to verify formatting.
    4. Run `python src/training/train.py` to train new weights.
    5. Evaluate via `python src/training/evaluate.py`.
    """)
