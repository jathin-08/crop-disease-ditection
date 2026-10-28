"""
Tomato Disease AI - Computer Vision Visualization Utility
Provides bounding box rendering, HUD overlays, pill badges, and disease advisory dictionary.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Comprehensive Tomato Disease Advisory Knowledge Base
TOMATO_DISEASE_INFO: Dict[str, Dict[str, Any]] = {
    "tomato_healthy": {
        "display_name": "Tomato Healthy Leaf",
        "scientific_name": "Solanum lycopersicum (Healthy)",
        "severity": "None",
        "description": "Vibrant, uniformly green foliage with no visible pathogen lesions, chlorosis, or necrotic spots.",
        "symptoms": "Uniform green coloration, healthy leaf architecture, vigorous turgor, no spots or fungal growth.",
        "organic_treatment": "Apply compost tea, vermicompost extract, and balanced seaweed liquid fertilizer to sustain foliage vigor.",
        "chemical_treatment": "No chemical treatment needed. Maintain standard balanced N-P-K fertilizer regimen.",
        "prevention": "Ensure well-draining soil, adequate sunlight (6-8 hrs/day), and proper drip irrigation spacing.",
        "color_bgr": (46, 184, 46)  # Bright Green
    },
    "tomato_bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "scientific_name": "Xanthomonas perforans / vesicatoria",
        "severity": "High",
        "description": "Destructive bacterial foliage disease causing circular brown/black spots with yellow halos.",
        "symptoms": "Small (2-3 mm) dark, water-soaked circular lesions on leaves, often surrounded by conspicuous yellow halos; foliage blights and drops prematurely.",
        "organic_treatment": "Spray fixed copper hydroxide or copper octanoate combined with Bacillus amyloliquefaciens biopesticide.",
        "chemical_treatment": "Copper fungicides tank-mixed with Mancozeb (protectant) applied at 7-10 day intervals during wet conditions.",
        "prevention": "Use certified disease-free seeds, avoid overhead sprinkler irrigation, practice a 2-year crop rotation.",
        "color_bgr": (0, 69, 255)  # Crimson Red
    },
    "tomato_early_blight": {
        "display_name": "Tomato Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "High",
        "description": "Fungal infection characterized by prominent target-board concentric ring lesions on older leaves.",
        "symptoms": "Dark brown to black necrotic spots with characteristic concentric rings ('bullseye' or target board pattern), surrounded by chlorotic yellow halos.",
        "organic_treatment": "Apply potassium bicarbonate, neem oil, copper soaps, and prune infected lower foliage from the ground up.",
        "chemical_treatment": "Apply Chlorothalonil, Azoxystrobin, or Difenoconazole at the first sign of symptoms.",
        "prevention": "Mulch heavily to prevent soil splashback, space plants for airflow, practice 3-year solanaceous crop rotation.",
        "color_bgr": (0, 140, 255)  # Deep Orange
    },
    "tomato_late_blight": {
        "display_name": "Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "Critical",
        "description": "Aggressive, fast-spreading oomycete water mold disease capable of destroying fields in days.",
        "symptoms": "Large, irregular, water-soaked greasy olive-green to dark brown patches; white fuzzy mold on leaf undersides in high humidity.",
        "organic_treatment": "Copper sulfate / Bordeaux mixture at early onset; rogue out and destroy severely blighted plants immediately.",
        "chemical_treatment": "Mandipropamid, Cymoxanil, or Dimethomorph systemic fungicides formulated for late blight control.",
        "prevention": "Eliminate cull piles, destroy volunteer tomato/potato plants, utilize resistant hybrid varieties (e.g., Defiant, Mountain Merit).",
        "color_bgr": (20, 20, 220)  # Scarlet
    },
    "tomato_leaf_mold": {
        "display_name": "Tomato Leaf Mold",
        "scientific_name": "Passalora fulva (Cladosporium fulvum)",
        "severity": "Moderate",
        "description": "Fungal disease common in high-humidity greenhouse and tunnel environments.",
        "symptoms": "Pale yellow chlorotic blotches on the upper leaf surface matching olive-green to velvety brown mold felt on the underside.",
        "organic_treatment": "Bio-fungicides containing Trichoderma harzianum or Bacillus subtilis; ventilate greenhouse to reduce relative humidity below 85%.",
        "chemical_treatment": "Fungicide sprays with Chlorothalonil, Mancozeb, or Boscalid.",
        "prevention": "Maximize air circulation, prune lower canopy suckers, utilize drip lines, grow mold-resistant cultivars.",
        "color_bgr": (0, 215, 255)  # Gold Yellow
    },
    "tomato_septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "severity": "Moderate",
        "description": "Common foliar fungal infection producing numerous small circular spots with distinct grey centers.",
        "symptoms": "Abundant small (1.5-3 mm) circular spots with dark brown margins and light tan-to-grey centers; leaves turn completely yellow and wither.",
        "organic_treatment": "Copper ammonium complex, sulfur sprays; strip infected bottom leaves up to the first fruit cluster.",
        "chemical_treatment": "Preventive sprays of Chlorothalonil or Mancozeb every 7-14 days.",
        "prevention": "Stake and cage plants, stake indeterminate stems, clean up all garden debris in autumn, sanitize tools.",
        "color_bgr": (255, 128, 0)  # Bright Blue / Sky
    },
    "tomato_spider_mites": {
        "display_name": "Tomato Spider Mites (Two-Spotted)",
        "scientific_name": "Tetranychus urticae",
        "severity": "Moderate",
        "description": "Microscopic arachnid sap-feeders causing speckled chlorosis, bronzing, and webbing.",
        "symptoms": "Fine yellow or white stippling (pinprick dots) on leaf surfaces; heavy infestations exhibit fine silken webs and leaf bronzing.",
        "organic_treatment": "Release predatory mites (Phytoseiulus persimilis); spray cold-pressed neem oil or potassium salts of fatty acids (insecticidal soap).",
        "chemical_treatment": "Abamectin, Bifenazate, or Spiromesifen miticides applied thoroughly under foliage.",
        "prevention": "Avoid drought stress, maintain moderate canopy humidity, keep dust down on field borders.",
        "color_bgr": (180, 105, 255)  # Violet
    },
    "tomato_target_spot": {
        "display_name": "Tomato Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "severity": "High",
        "description": "Fungal pathogen causing brown lesions with pale concentric rings on leaves and fruit stems.",
        "symptoms": "Brown circular to irregular necrotic lesions with faint concentric rings; lesions coalesce causing extensive foliage collapse.",
        "organic_treatment": "Bio-fungicides based on Bacillus subtilis; prompt disposal of infected plant material.",
        "chemical_treatment": "Pyraclostrobin, Fluxapyroxad, or Azoxystrobin fungicides.",
        "prevention": "Ensure wide row spacing (minimum 1m), avoid wet foliage during nightfall, remove solanaceous weed hosts.",
        "color_bgr": (75, 0, 130)  # Indigo
    },
    "tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus (ToMV)",
        "scientific_name": "Tobamovirus (ToMV)",
        "severity": "Critical",
        "description": "Extremely contagious, mechanically transmitted plant virus persisting in soil and seed coats.",
        "symptoms": "Mottled light and dark green mosaic patterns on leaves, 'shoestring' or fern-like leaf distortion, stunted growth, mottled fruit.",
        "organic_treatment": "No cure once infected. Rogue and burn or bag infected plants; spray milk solution (20%) during handling to inhibit viral transfer.",
        "chemical_treatment": "No virucidal chemical exists. Disinfect greenhouse benches and shears with 10% trisodium phosphate (TSP) or 20% bleach.",
        "prevention": "Wash hands with soap and water after handling tobacco products, plant certified virus-free seed, sterilize greenhouse stakes.",
        "color_bgr": (128, 0, 128)  # Purple
    },
    "tomato_yellow_leaf_curl_virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "scientific_name": "Begomovirus (TYLCV)",
        "severity": "Critical",
        "description": "Devastating viral disease transmitted primarily by the sweetpotato whitefly (Bemisia tabaci).",
        "symptoms": "Marked upward cupping and curling of leaf margins, bright interveinal chlorosis (yellowing), severe plant stunting and flower abortion.",
        "organic_treatment": "Control whitefly vectors using yellow sticky card traps, reflective silver mulches, and botanical pyrethrins.",
        "chemical_treatment": "Systemic insecticides (Imidacloprid, Thiamethoxam, or Dinotefuran) directed against whitefly vectors.",
        "prevention": "Install 50-mesh insect-proof netting in greenhouses, plant TYLCV-resistant hybrids, control weed hosts around borders.",
        "color_bgr": (0, 215, 255)  # Cyan Gold
    }
}


def get_disease_info(class_name: str) -> Dict[str, Any]:
    """Retrieves full advisory metadata for a disease class."""
    clean_name = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    
    # Direct match
    if clean_name in TOMATO_DISEASE_INFO:
        return TOMATO_DISEASE_INFO[clean_name]

    # Partial match
    for k, v in TOMATO_DISEASE_INFO.items():
        if clean_name in k or k in clean_name:
            return v

    # Fallback generic info
    return {
        "display_name": class_name.replace("_", " ").title(),
        "scientific_name": "Solanum lycopersicum condition",
        "severity": "Moderate",
        "description": "Foliage condition detected on tomato leaf.",
        "symptoms": "Unusual leaf discoloration or texture variation.",
        "organic_treatment": "Prune symptomatic leaflets and apply broad-spectrum organic copper fungicide.",
        "chemical_treatment": "Broad-spectrum agricultural protective fungicide.",
        "prevention": "Ensure good airflow and avoid wet foliage.",
        "color_bgr": (0, 200, 255)
    }


def draw_rounded_rectangle(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    radius: int = 8,
    fill: bool = False
) -> None:
    """Draws a rounded rectangle on an OpenCV image."""
    if cv2 is None or img is None:
        return

    x1, y1 = pt1
    x2, y2 = pt2
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    radius = min(radius, w // 2, h // 2)
    if radius <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1 if fill else thickness)
        return

    if fill:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def annotate_detection(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    track_id: Optional[int] = None,
    iou: Optional[float] = None
) -> np.ndarray:
    """Renders high-contrast bounding box with corner brackets and pill badge."""
    if cv2 is None or frame is None:
        return frame

    x1, y1, x2, y2 = map(int, box)
    info = get_disease_info(class_name)
    color = info.get("color_bgr", (0, 215, 255))

    # Main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Corner brackets for premium visual appearance
    corner_len = max(12, int(min(x2 - x1, y2 - y1) * 0.15))
    accent = (255, 255, 255)
    thick = 3
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), accent, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), accent, thick)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), accent, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), accent, thick)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), accent, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), accent, thick)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), accent, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), accent, thick)

    # Label text
    id_tag = f"#{track_id} " if track_id is not None else ""
    if iou is not None:
        label = f"{id_tag}{info['display_name']} ({confidence * 100:.1f}%) | IoU: {iou:.2f}"
    else:
        label = f"{id_tag}{info['display_name']} ({confidence * 100:.1f}%)"
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.52
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

    badge_h = th + 10
    badge_w = tw + 16
    badge_y1 = max(0, y1 - badge_h - 4)
    badge_y2 = badge_y1 + badge_h
    badge_x1 = x1
    badge_x2 = min(frame.shape[1], x1 + badge_w)

    overlay = frame.copy()
    draw_rounded_rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), color, fill=True, radius=6)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, label, (badge_x1 + 8, badge_y1 + th + 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud_card(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    fps: Optional[float] = None
) -> np.ndarray:
    """Overlays an aesthetic, semi-transparent HUD summary banner on the top edge."""
    if cv2 is None or frame is None:
        return frame

    h, w = frame.shape[:2]
    hud_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 24, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, hud_h), (w, hud_h), (46, 184, 46), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Left brand
    cv2.putText(frame, "TOMATO LEAF AI | YOLO26", (15, 28), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Center detections count
    n_det = len(detections)
    if n_det > 0:
        det_text = f"Detections: {n_det} | Top: {detections[0]['disease_name']} ({detections[0]['confidence']*100:.1f}%)"
        cv2.putText(frame, det_text, (280, 28), font, 0.50, (100, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Status: Monitoring Foliage (No Lesions Detected)", (280, 28), font, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Right FPS
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 110, 28), font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    return frame
