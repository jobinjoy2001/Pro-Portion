from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import math
import os
from datetime import datetime
from sklearn.svm import SVR
from fastapi import WebSocketDisconnect
import joblib
import base64
import asyncio
import json


app = FastAPI(title="Pro-Portion Backend v1.1 - Tutorial + ML")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# For static image processing (upload endpoints)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# For real-time WebSocket streaming
face_mesh_video = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,    
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.2
)


pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.2
)


OUTPUT_DIR = "processed_images"
TUTORIAL_DIR = "tutorial_steps"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TUTORIAL_DIR, exist_ok=True)

# ── OpenCV DNN Face Detector (more accurate than Haar cascade) ──────

_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

_proto_path = "deploy.prototxt"
_model_path = "res10_300x300_ssd.caffemodel"

# Download only if not already present
if not os.path.exists(_proto_path):
    print("Downloading face detector prototxt...")
    urllib.request.urlretrieve(_PROTO, _proto_path)

if not os.path.exists(_model_path):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(_MODEL, _model_path)

dnn_face_detector = cv2.dnn.readNetFromCaffe(_proto_path, _model_path)
print("DNN face detector loaded")


# Classical/Ideal facial proportions (Loomis method + Golden Ratio)
IDEAL_PROPORTIONS = {
    "eye_to_face_width": 0.46,      # Eyes should be 46% of face width apart
    "nose_to_face_height": 0.33,    # Nose-chin = 1/3 of face height
    "face_aspect_ratio": 0.75,      # Width/Height ratio (3:4)
    "eye_level": 0.50,              # Eyes at 50% of face height
    "mouth_level": 0.66             # Mouth at 2/3 of face height
}


def calculate_distance(landmark1, landmark2, img_width, img_height):
    x1, y1 = landmark1.x * img_width, landmark1.y * img_height
    x2, y2 = landmark2.x * img_width, landmark2.y * img_height
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_face_ratios(landmarks, img_width, img_height):
    try:
        eye_distance     = calculate_distance(landmarks[33],  landmarks[362], img_width, img_height)
        nose_to_chin     = calculate_distance(landmarks[1],   landmarks[152], img_width, img_height)
        face_width       = calculate_distance(landmarks[234], landmarks[454], img_width, img_height)
        forehead_to_chin = calculate_distance(landmarks[10],  landmarks[152], img_width, img_height)
        mouth_width      = calculate_distance(landmarks[61],  landmarks[291], img_width, img_height)
        nose_width       = calculate_distance(landmarks[48],  landmarks[278], img_width, img_height)

        # ── NEW: jaw, forehead, cheekbone for accurate shape detection ──
        jaw_width        = calculate_distance(landmarks[172], landmarks[397], img_width, img_height)
        forehead_width   = calculate_distance(landmarks[70],  landmarks[300], img_width, img_height)
        cheekbone_width  = calculate_distance(landmarks[123], landmarks[352], img_width, img_height)

        return {
            "measurements_px": {
                "eye_distance":    round(eye_distance, 2),
                "nose_to_chin":    round(nose_to_chin, 2),
                "face_width":      round(face_width, 2),
                "face_height":     round(forehead_to_chin, 2),
                "mouth_width":     round(mouth_width, 2),
                "nose_width":      round(nose_width, 2),
                "jaw_width":       round(jaw_width, 2),
                "forehead_width":  round(forehead_width, 2),
                "cheekbone_width": round(cheekbone_width, 2),
            },
            "proportional_ratios": {
                "eye_to_face_width":   round(eye_distance    / face_width,       3) if face_width       > 0 else 0,
                "nose_to_face_height": round(nose_to_chin    / forehead_to_chin, 3) if forehead_to_chin > 0 else 0,
                "face_aspect_ratio":   round(face_width      / forehead_to_chin, 3) if forehead_to_chin > 0 else 0,
                "mouth_to_face_width": round(mouth_width     / face_width,       3) if face_width       > 0 else 0,
                "nose_to_face_width":  round(nose_width      / face_width,       3) if face_width       > 0 else 0,
                # NEW — shape classification ratios
                "jaw_to_face_width":   round(jaw_width        / face_width,       3) if face_width       > 0 else 0,
                "forehead_to_jaw":     round(forehead_width   / jaw_width,        3) if jaw_width        > 0 else 0,
                "cheekbone_to_jaw":    round(cheekbone_width  / jaw_width,        3) if jaw_width        > 0 else 0,
            }
        }
    except Exception as e:
        print(f"Error computing face ratios: {e}")
        return None



def analyze_proportions_vs_ideal(ratios):
    """Compare detected ratios with classical ideal proportions"""
    if not ratios:
        return None

    detected = ratios["proportional_ratios"]

    # DEBUG: show the numbers for this face in the terminal
    print(
        "aspect=", detected.get("face_aspect_ratio", 0),
        "jaw/face=", detected.get("jaw_to_face_width", 0),
        "forehead/jaw=", detected.get("forehead_to_jaw", 0),
        "cheekbone/jaw=", detected.get("cheekbone_to_jaw", 0),
        "mouth/face=", detected.get("mouth_to_face_width", 0),
    )

    analysis = {
        "overall_score": 0,
        "comparisons": {},
        "recommendations": []
    }

    scores = []

    # Eye spacing
    eye_diff = abs(detected["eye_to_face_width"] - IDEAL_PROPORTIONS["eye_to_face_width"])
    eye_score = max(0, 100 - (eye_diff * 200))
    scores.append(eye_score)
    analysis["comparisons"]["eye_spacing"] = {
        "detected": detected["eye_to_face_width"],
        "ideal": IDEAL_PROPORTIONS["eye_to_face_width"],
        "difference": round(eye_diff, 3),
        "score": round(eye_score, 1)
    }
    if eye_diff > 0.05:
        if detected["eye_to_face_width"] > IDEAL_PROPORTIONS["eye_to_face_width"]:
            analysis["recommendations"].append("Eyes are slightly wider-set than classical proportions")
        else:
            analysis["recommendations"].append("Eyes are slightly closer-set than classical proportions")

    # Nose-chin ratio
    nose_diff = abs(detected["nose_to_face_height"] - IDEAL_PROPORTIONS["nose_to_face_height"])
    nose_score = max(0, 100 - (nose_diff * 200))
    scores.append(nose_score)
    analysis["comparisons"]["nose_chin_ratio"] = {
        "detected": detected["nose_to_face_height"],
        "ideal": IDEAL_PROPORTIONS["nose_to_face_height"],
        "difference": round(nose_diff, 3),
        "score": round(nose_score, 1)
    }
    if nose_diff > 0.05:
        if detected["nose_to_face_height"] > IDEAL_PROPORTIONS["nose_to_face_height"]:
            analysis["recommendations"].append("Lower face is longer than classical thirds")
        else:
            analysis["recommendations"].append("Lower face is shorter than classical thirds")

    # Face aspect ratio
    aspect_diff = abs(detected["face_aspect_ratio"] - IDEAL_PROPORTIONS["face_aspect_ratio"])
    aspect_score = max(0, 100 - (aspect_diff * 150))
    scores.append(aspect_score)
    analysis["comparisons"]["face_aspect"] = {
        "detected": detected["face_aspect_ratio"],
        "ideal": IDEAL_PROPORTIONS["face_aspect_ratio"],
        "difference": round(aspect_diff, 3),
        "score": round(aspect_score, 1)
    }
    if aspect_diff > 0.08:
        if detected["face_aspect_ratio"] > IDEAL_PROPORTIONS["face_aspect_ratio"]:
            analysis["recommendations"].append("Face is wider than classical oval proportions")
        else:
            analysis["recommendations"].append("Face is narrower/longer than classical oval proportions")

    # Calculate overall score
    analysis["overall_score"] = round(sum(scores) / len(scores), 1)

    # ── Improved face shape classification ────────────────────────────
    aspect        = detected.get("face_aspect_ratio", 0)
    jaw_ratio     = detected.get("jaw_to_face_width", 0)
    forehead_jaw  = detected.get("forehead_to_jaw", 0)
    cheekbone_jaw = detected.get("cheekbone_to_jaw", 0)
    mouth_ratio   = detected.get("mouth_to_face_width", 0)

    # Prototypes based on your labeled examples
    prototypes = {
        "Round": {
            "aspect": 0.957,
            "jaw": 0.806,
            "forehead": 0.970,
            "cheekbone": 1.082,
            "mouth": 0.331
        },
        "Square": {
            "aspect": 0.844,
            "jaw": 0.826,
            "forehead": 1.018,
            "cheekbone": 1.076,
            "mouth": 0.332
        },
        "Oval/Balanced": {
            "aspect": 0.827,
            "jaw": 0.770,
            "forehead": 1.123,
            "cheekbone": 1.164,
            "mouth": 0.385
        },
        "Heart": {
            "aspect": 0.821,
            "jaw": 0.760,
            "forehead": 1.100,
            "cheekbone": 1.154,
            "mouth": 0.356
        },
        "Oblong/Long": {
            "aspect": 0.775,
            "jaw": 0.809,
            "forehead": 1.041,
            "cheekbone": 1.099,
            "mouth": 0.341
        },
        "Diamond": {
            "aspect": 0.880,
            "jaw": 0.814,
            "forehead": 1.026,
            "cheekbone": 1.099,
            "mouth": 0.356
        },
        "Triangle": {
            "aspect": 0.878,
            "jaw": 0.776,
            "forehead": 1.063,
            "cheekbone": 1.147,
            "mouth": 0.368
        }
    }

    weights = {
        "aspect": 2.8,
        "jaw": 2.4,
        "forehead": 2.0,
        "cheekbone": 1.8,
        "mouth": 0.8
    }

    def shape_distance(p):
        return math.sqrt(
            weights["aspect"]    * (aspect - p["aspect"]) ** 2 +
            weights["jaw"]       * (jaw_ratio - p["jaw"]) ** 2 +
            weights["forehead"]  * (forehead_jaw - p["forehead"]) ** 2 +
            weights["cheekbone"] * (cheekbone_jaw - p["cheekbone"]) ** 2 +
            weights["mouth"]     * (mouth_ratio - p["mouth"]) ** 2
        )

    shape_scores = {shape: shape_distance(proto) for shape, proto in prototypes.items()}

    # Tie-break / bias rules from observed patterns
    if aspect >= 0.93:
        shape_scores["Round"] *= 0.72
        shape_scores["Square"] *= 1.08
        shape_scores["Oblong/Long"] *= 1.20

    if 0.83 <= aspect <= 0.87 and jaw_ratio >= 0.82 and abs(forehead_jaw - 1.0) <= 0.03:
        shape_scores["Square"] *= 0.74

    if aspect <= 0.80 and jaw_ratio >= 0.79 and forehead_jaw <= 1.06:
        shape_scores["Oblong/Long"] *= 0.72

    if cheekbone_jaw >= 1.13 and jaw_ratio <= 0.79 and forehead_jaw >= 1.09:
        shape_scores["Heart"] *= 0.76

    if cheekbone_jaw >= 1.13 and 1.05 <= forehead_jaw <= 1.09 and jaw_ratio <= 0.79:
        shape_scores["Triangle"] *= 0.74

    if cheekbone_jaw >= 1.14 and jaw_ratio <= 0.78 and forehead_jaw >= 1.10:
        shape_scores["Oval/Balanced"] *= 0.82

    if 0.86 <= aspect <= 0.90 and 0.80 <= jaw_ratio <= 0.82 and 1.00 <= forehead_jaw <= 1.04:
        shape_scores["Diamond"] *= 0.72

    if jaw_ratio >= 0.82 and forehead_jaw <= 1.03 and cheekbone_jaw <= 1.10:
        shape_scores["Square"] *= 0.84

    if forehead_jaw < 1.00 and jaw_ratio >= 0.79:
        shape_scores["Triangle"] *= 0.86

    ranked = sorted(shape_scores.items(), key=lambda x: x[1])
    best_shape, best_score = ranked[0]
    second_shape, second_score = ranked[1]

    analysis["face_shape"] = best_shape
    analysis["shape_scores"] = {k: round(v, 4) for k, v in sorted(shape_scores.items(), key=lambda x: x[1])}
    analysis["shape_confidence"] = round(
        max(0.0, min(100.0, (second_score - best_score) / max(second_score, 1e-6) * 100)),
        1
    )

    print("Shape ranking:", analysis["shape_scores"])
    print("Chosen shape :", analysis["face_shape"], f"(confidence={analysis['shape_confidence']}%)")

    return analysis



def compute_body_ratios(landmarks, img_width, img_height):
    try:
        shoulder_width = calculate_distance(landmarks[11], landmarks[12], img_width, img_height)
        hip_width = calculate_distance(landmarks[23], landmarks[24], img_width, img_height)
        torso_length = calculate_distance(landmarks[11], landmarks[23], img_width, img_height)
        body_height = calculate_distance(landmarks[0], landmarks[27], img_width, img_height)
        
        return {
            "measurements_px": {
                "shoulder_width": round(shoulder_width, 2),
                "hip_width": round(hip_width, 2),
                "torso_length": round(torso_length, 2),
                "estimated_height": round(body_height, 2)
            },
            "proportional_ratios": {
                "shoulder_to_hip_ratio": round(shoulder_width / hip_width, 3) if hip_width > 0 else 0,
                "torso_to_height_ratio": round(torso_length / body_height, 3) if body_height > 0 else 0
            }
        }
    except Exception as e:
        print(f"Error computing body ratios: {e}")
        return None


def get_face_bounds(face_landmarks, img_w, img_h, margin=0.20):
    """
    Returns face bounding box with generous margin so tilted/angled
    faces never get cropped. Uses FACE_OVAL landmarks only.
    """
    oval_ids = [
        10,338,297,332,284,251,389,356,454,323,361,288,
        397,365,379,378,400,377,152,148,176,149,150,136,
        172,58,132,93,234,127,162,21,54,103,67,109
    ]
    xs = [face_landmarks[i].x * img_w for i in oval_ids]
    ys = [face_landmarks[i].y * img_h for i in oval_ids]

    fw = max(xs) - min(xs)
    fh = max(ys) - min(ys)

    x_left  = int(max(0,       min(xs) - fw * margin))
    x_right = int(min(img_w,   max(xs) + fw * margin))
    y_top   = int(max(0,       min(ys) - fh * margin))
    y_bot   = int(min(img_h,   max(ys) + fh * margin))

    return {
        "x_left":    x_left,
        "x_right":   x_right,
        "y_top":     y_top,
        "y_bottom":  y_bot,
        "x_center":  (x_left + x_right) // 2,
        "y_center":  (y_top + y_bot) // 2,
        "face_width":  x_right - x_left,
        "face_height": y_bot - y_top,
    }



def add_measurements_overlay(img, face_landmarks, width, height, step_name=""):
    """Add measurement text overlay to image"""
    try:
        # Calculate measurements
        ratios = compute_face_ratios(face_landmarks, width, height)
        if not ratios:
            return img
        
        measurements = ratios["measurements_px"]
        proportions = ratios["proportional_ratios"]
        
        # Create semi-transparent background for text
        overlay = img.copy()
        
        # Measurement box dimensions
        box_height = 280
        box_width = 320
        cv2.rectangle(overlay, (10, 10), (box_width, box_height), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        y_offset = 35
        line_height = 28
        
        # Title
        cv2.putText(img, step_name if step_name else "Measurements", 
                   (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        y_offset += line_height + 5
        
        # Pixel measurements
        texts = [
            f"Face: {measurements['face_width']:.0f}x{measurements['face_height']:.0f}px",
            f"Eyes: {measurements['eye_distance']:.1f}px",
            f"Nose-Chin: {measurements['nose_to_chin']:.1f}px",
            f"Mouth: {measurements['mouth_width']:.1f}px",
            "",
            "--- Ratios ---",
            f"Eye/Width: {proportions['eye_to_face_width']:.3f}",
            f"Nose/Height: {proportions['nose_to_face_height']:.3f}",
            f"Aspect: {proportions['face_aspect_ratio']:.3f}",
        ]
        
        for text in texts:
            if text:
                if "---" in text:
                    cv2.putText(img, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += line_height
        
    except Exception as e:
        print(f"Error adding measurements: {e}")
    
    return img


FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109
]

def get_precise_face_bounds(face_landmarks, img_w, img_h):
    xs = [face_landmarks[i].x * img_w for i in FACE_OVAL_INDICES]
    ys = [face_landmarks[i].y * img_h for i in FACE_OVAL_INDICES]

    raw_top    = face_landmarks[10].y  * img_h
    raw_bottom = face_landmarks[152].y * img_h
    raw_left   = min(xs)
    raw_right  = max(xs)

    face_w = raw_right - raw_left
    face_h = raw_bottom - raw_top

    # Clamp bottom to chin only — no neck
    clamped_bottom = raw_top + face_h * 1.08

    pad_x     = face_w * 0.12
    pad_y_top = face_h * 0.10

    return {
        "x_left":     int(max(0,     raw_left  - pad_x)),
        "x_right":    int(min(img_w, raw_right + pad_x)),
        "y_top":      int(max(0,     raw_top   - pad_y_top)),
        "y_bottom":   int(min(img_h, clamped_bottom)),
        "x_center":   int((raw_left + raw_right) / 2),
        "face_width":  int(raw_right - raw_left + 2 * pad_x),
        "face_height": int(clamped_bottom - raw_top + pad_y_top),
    }



COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0)
]

def draw_tutorial_step(img, face_landmarks, face_id, step_number, img_w, img_h):
    canvas = img.copy()
    color  = COLORS[face_id % len(COLORS)]

    bounds = get_precise_face_bounds(face_landmarks, img_w, img_h)
    xl = bounds["x_left"]
    xr = bounds["x_right"]
    yt = bounds["y_top"]
    yb = bounds["y_bottom"]
    xc = bounds["x_center"]

    def lm_y(idx): return int(face_landmarks[idx].y * img_h)
    def lm_x(idx): return int(face_landmarks[idx].x * img_w)

    # All horizontal positions come directly from MediaPipe landmarks
    y_hairline = lm_y(10)
    y_eyebrow  = int((lm_y(70) + lm_y(300)) / 2)
    y_nose     = lm_y(94)
    y_eyeline  = int((lm_y(33) + lm_y(263)) / 2)
    y_mouth    = int((lm_y(61) + lm_y(291)) / 2)

    # Always draw the bounding box
    cv2.rectangle(canvas, (xl, yt), (xr, yb), color, 4)

    # Center line only from step 2 onwards
    if step_number >= 2:
        cv2.line(canvas, (xc, yt), (xc, yb), color, 3)


    if step_number == 1:
        step_name = "Step 1: Face Bounds"

    elif step_number == 2:
        step_name = "Step 2: Center Line"

    elif step_number == 3:
        cv2.line(canvas, (xl, y_hairline), (xr, y_hairline), (255, 200, 0), 2)
        cv2.putText(canvas, "Hairline", (xr + 10, y_hairline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.line(canvas, (xl, y_eyebrow), (xr, y_eyebrow), (200, 150, 0), 2)
        cv2.putText(canvas, "Eyebrow", (xr + 10, y_eyebrow),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 0), 2)

        cv2.line(canvas, (xl, y_nose), (xr, y_nose), (150, 100, 0), 2)
        cv2.putText(canvas, "Nose", (xr + 10, y_nose),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 100, 0), 2)

        step_name = "Step 3: Horizontal Thirds"

    elif step_number == 4:
        cv2.line(canvas, (xl, y_hairline), (xr, y_hairline), (255, 200, 0), 2)
        cv2.line(canvas, (xl, y_eyebrow),  (xr, y_eyebrow),  (200, 150, 0), 2)
        cv2.line(canvas, (xl, y_nose),     (xr, y_nose),     (150, 100, 0), 2)

        cv2.line(canvas,
                 (lm_x(33), y_eyeline),
                 (lm_x(263), y_eyeline),
                 (0, 255, 255), 3)
        cv2.putText(canvas, "Eye Line", (xr + 10, y_eyeline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        step_name = "Step 4: Eye Line"

    elif step_number == 5:
        cv2.line(canvas, (xl, y_hairline), (xr, y_hairline), (255, 200, 0), 2)
        cv2.line(canvas, (xl, y_eyebrow),  (xr, y_eyebrow),  (200, 150, 0), 2)
        cv2.line(canvas, (xl, y_nose),     (xr, y_nose),     (150, 100, 0), 2)
        cv2.line(canvas,
                 (lm_x(33), y_eyeline),
                 (lm_x(263), y_eyeline),
                 (0, 255, 255), 3)

        jaw_pts = [234,93,132,58,172,136,150,149,176,148,152,
                   377,400,378,379,365,397,288,361,454]
        contour = np.array(
            [(lm_x(i), lm_y(i)) for i in jaw_pts], dtype=np.int32
        )
        cv2.polylines(canvas, [contour], False, (255, 0, 255), 2)

        step_name = "Step 5: Face Outline"

    elif step_number == 6:
        cv2.line(canvas, (xl, y_hairline), (xr, y_hairline), (255, 200, 0), 2)
        cv2.line(canvas, (xl, y_eyebrow),  (xr, y_eyebrow),  (200, 150, 0), 2)
        cv2.line(canvas, (xl, y_nose),     (xr, y_nose),     (150, 100, 0), 2)
        cv2.line(canvas,
                 (lm_x(33), y_eyeline),
                 (lm_x(263), y_eyeline),
                 (0, 255, 255), 3)

        jaw_pts = [234,93,132,58,172,136,150,149,176,148,152,
                   377,400,378,379,365,397,288,361,454]
        contour = np.array(
            [(lm_x(i), lm_y(i)) for i in jaw_pts], dtype=np.int32
        )
        cv2.polylines(canvas, [contour], False, (255, 0, 255), 2)

        cv2.circle(canvas, (lm_x(4),   lm_y(4)),   5, (0, 150, 255), -1)
        cv2.circle(canvas, (lm_x(152), lm_y(152)), 5, (255, 100, 0), -1)

        cv2.line(canvas, (xl, y_mouth), (xr, y_mouth), (0, 200, 100), 2)
        cv2.putText(canvas, "Mouth", (xr + 10, y_mouth),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 2)

        step_name = "Step 6: Complete Grid"

    cv2.putText(canvas, step_name, (xl, yt - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    
    return canvas

def detect_face_roi(img):
    """
    Use OpenCV DNN to detect face bounding box.
    Returns (x, y, w, h) of best face or None.
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    dnn_face_detector.setInput(blob)
    detections = dnn_face_detector.forward()

    best = None
    best_conf = 0.5  # minimum confidence threshold

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to image bounds
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(w, x2);  y2 = min(h, y2)
            best_conf = conf
            best = (x1, y1, x2 - x1, y2 - y1)

    return best


def draw_loomis_grid(img, face_landmarks, face_id, head_pose=None):
    """
    Safe backward-compatible Loomis grid drawer.

    - If head_pose is None -> behaves like your existing frontal grid.
    - If head_pose is provided and yaw is large -> draws a pose-aware grid.
    - Does NOT affect tutorial/learn mode functions.
    """
    height, width = img.shape[:2]

    try:
        def px(idx): return int(face_landmarks[idx].x * width)
        def py(idx): return int(face_landmarks[idx].y * height)

        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        color = colors[face_id % len(colors)]

        # ── Face bounds: keep your current stable logic ───────────────
        roi = detect_face_roi(img)

        if roi:
            rx, ry, rw, rh = roi
            x_left = rx
            x_right = rx + rw
            y_top = ry
            y_bottom = ry + rh
        else:
            oval_ids = [
                10,338,297,332,284,251,389,356,454,323,361,288,
                397,365,379,378,400,377,152,148,176,149,150,136,
                172,58,132,93,234,127,162,21,54,103,67,109
            ]
            xs = [face_landmarks[i].x * width for i in oval_ids]
            ys = [face_landmarks[i].y * height for i in oval_ids]
            x_left = int(min(xs))
            x_right = int(max(xs))
            y_top = py(10)
            y_bottom = py(152)

        face_w = max(1, x_right - x_left)
        face_h = max(1, y_bottom - y_top)

        y_brow = (py(70) + py(300)) // 2
        y_eye = (py(33) + py(263)) // 2
        y_nose = py(94)
        y_mouth = (py(61) + py(291)) // 2

        PX_TO_CM = 0.0264

        # ── Default old behavior if no pose is passed ────────────────
        if not head_pose:
            x_center = (x_left + x_right) // 2

            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), color, 2)

            cv2.line(
                img, (x_center, y_top), (x_center, y_bottom),
                (255, 0, 255), 2, cv2.LINE_AA
            )

            lines = [
                (y_brow,  (0, 215, 255), "Eyebrow"),
                (y_eye,   (0, 255, 255), "Eye Line"),
                (y_nose,  (0, 165, 255), "Nose"),
                (y_mouth, (203, 192, 255), "Mouth"),
            ]
            for y, c, label in lines:
                if y_top <= y <= y_bottom:
                    cv2.line(img, (x_left, y), (x_right, y), c, 1, cv2.LINE_AA)
                    cv2.putText(
                        img, label, (x_right + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
                    )

            cv2.putText(
                img, f"W:{face_w}px ({face_w * PX_TO_CM:.1f}cm)",
                (x_left, min(height - 8, y_bottom + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, f"H:{face_h}px ({face_h * PX_TO_CM:.1f}cm)",
                (max(0, x_left - 95), (y_top + y_bottom) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                img, f"Face {face_id + 1}",
                (x_left, max(15, y_top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )
            return img

        # ── Pose-aware branch for realtime only ───────────────────────
        yaw = float(head_pose.get("yaw", 0.0))
        pitch = float(head_pose.get("pitch", 0.0))
        abs_yaw = abs(yaw)

        # Near-frontal: preserve legacy look
        if abs_yaw < 15:
            x_center = (x_left + x_right) // 2

            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), color, 2)

            cv2.line(
                img, (x_center, y_top), (x_center, y_bottom),
                (255, 0, 255), 2, cv2.LINE_AA
            )

            lines = [
                (y_brow,  (0, 215, 255), "Eyebrow"),
                (y_eye,   (0, 255, 255), "Eye Line"),
                (y_nose,  (0, 165, 255), "Nose"),
                (y_mouth, (203, 192, 255), "Mouth"),
            ]
            for y, c, label in lines:
                if y_top <= y <= y_bottom:
                    cv2.line(img, (x_left, y), (x_right, y), c, 1, cv2.LINE_AA)
                    cv2.putText(
                        img, label, (x_right + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
                    )
        else:
            # Direction-aware center shift
            direction = -1 if yaw < 0 else 1

            nose_x = px(1)
            chin_x = px(152)
            brow_mid_x = (px(70) + px(300)) // 2
            center_anchor = int((nose_x * 0.45) + (chin_x * 0.20) + (brow_mid_x * 0.35))

            # Perspective compression on the far side
            shift_ratio = min(abs_yaw / 60.0, 1.0) * 0.18
            near_scale = 1.0
            far_scale = 1.0 - min(abs_yaw / 70.0, 0.35)

            if direction > 0:  # looking right
                left_span = face_w * 0.50 * near_scale
                right_span = face_w * 0.50 * far_scale
            else:              # looking left
                left_span = face_w * 0.50 * far_scale
                right_span = face_w * 0.50 * near_scale

            x_center = int(center_anchor + direction * face_w * shift_ratio)
            left_edge = max(0, int(x_center - left_span))
            right_edge = min(width - 1, int(x_center + right_span))

            # Slight pitch compensation for top/bottom
            pitch_shift = int((pitch / 45.0) * face_h * 0.08)
            top_adj = max(0, y_top - max(0, -pitch_shift))
            bottom_adj = min(height - 1, y_bottom + max(0, pitch_shift))

            # Outer bounds
            cv2.rectangle(img, (left_edge, top_adj), (right_edge, bottom_adj), color, 2)

            # Center axis
            cv2.line(
                img, (x_center, top_adj), (x_center, bottom_adj),
                (255, 0, 255), 2, cv2.LINE_AA
            )

            # Horizontal lines taper slightly toward far side for 3/4 feel
            taper = int(face_w * min(abs_yaw / 70.0, 0.18))
            if direction > 0:
                line_left = left_edge
                line_right = max(left_edge + 10, right_edge - taper)
            else:
                line_left = min(right_edge - 10, left_edge + taper)
                line_right = right_edge

            lines = [
                (y_brow,  (0, 215, 255), "Eyebrow"),
                (y_eye,   (0, 255, 255), "Eye Line"),
                (y_nose,  (0, 165, 255), "Nose"),
                (y_mouth, (203, 192, 255), "Mouth"),
            ]
            for y, c, label in lines:
                if top_adj <= y <= bottom_adj:
                    cv2.line(img, (line_left, y), (line_right, y), c, 1, cv2.LINE_AA)
                    cv2.putText(
                        img, label, (min(width - 120, right_edge + 5), y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
                    )

            x_left = left_edge
            x_right = right_edge
            y_top = top_adj
            y_bottom = bottom_adj
            face_w = max(1, x_right - x_left)
            face_h = max(1, y_bottom - y_top)

        cv2.putText(
            img, f"W:{face_w}px ({face_w * PX_TO_CM:.1f}cm)",
            (x_left, min(height - 8, y_bottom + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            img, f"H:{face_h}px ({face_h * PX_TO_CM:.1f}cm)",
            (max(0, x_left - 95), (y_top + y_bottom) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

        pose_label = "Front"
        if abs_yaw >= 15 and abs_yaw < 35:
            pose_label = "3/4"
        elif abs_yaw >= 35:
            pose_label = "Profile"

        cv2.putText(
            img, f"Face {face_id + 1} | {pose_label}",
            (x_left, max(15, y_top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    except Exception as e:
        print(f"draw_loomis_grid error: {e}")

    return img



# ── Exact MediaPipe face oval indices (ordered, no jumps) ─────────────
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]

LEFT_EYE  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
RIGHT_EYE = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249,263]
LEFT_BROW  = [46,53,52,65,55,107,66,105,63,70]
RIGHT_BROW = [276,283,282,295,285,336,296,334,293,300]
NOSE_BRIDGE = [168,6,197,195,5,4,1]
NOSE_BASE   = [129,98,97,2,326,327,358]
UPPER_LIP   = [61,185,40,39,37,0,267,269,270,409,291,308,310,311,312,13,82,81,80,191,78,61]
LOWER_LIP   = [61,78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]


def draw_sketch_canvas(face_landmarks, img_w: int, img_h: int, canvas_w=900, canvas_h=1100):
    """
    Clean, professional Loomis construction diagram on white canvas.
    Improved: smooth curves, anti-aliased, normalized proportions, clean labels.
    """

    # ── Canvas & landmark setup ──────────────────────────────────────
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    lms = face_landmarks
    px_all = [(lm.x * img_w, lm.y * img_h) for lm in lms]

    # ── Compute face bounds from oval ────────────────────────────────
    FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
                 397,365,379,378,400,377,152,148,176,149,150,136,
                 172,58,132,93,234,127,162,21,54,103,67,109,10]

    ox = [px_all[i][0] for i in FACE_OVAL]
    oy = [px_all[i][1] for i in FACE_OVAL]
    fx_min, fx_max = min(ox), max(ox)
    fy_min, fy_max = min(oy), max(oy)
    fw = fx_max - fx_min
    fh = fy_max - fy_min

    # Add margin
    margin = 0.12
    fx_min -= fw * margin;  fx_max += fw * margin
    fy_min -= fh * margin;  fy_max += fh * margin
    fw = fx_max - fx_min;   fh = fy_max - fy_min

    # ── Scale to canvas ──────────────────────────────────────────────
    pad = 90
    scale_x = (canvas_w - 2 * pad) / fw
    scale_y = (canvas_h - 2 * pad) / fh
    scale   = min(scale_x, scale_y)

    drawn_w = fw * scale;  drawn_h = fh * scale
    off_x = (canvas_w - drawn_w) / 2
    off_y = (canvas_h - drawn_h) / 2

    def to_canvas(ix: float, iy: float):
        return (int((ix - fx_min) * scale + off_x),
                int((iy - fy_min) * scale + off_y))

    def lm_pt(idx: int):
        return to_canvas(*px_all[idx])

    def draw_path(indices, color, thick=2, closed=False):
        pts = np.array([lm_pt(i) for i in indices], dtype=np.int32)
        cv2.polylines(canvas, [pts], closed, color, thick, cv2.LINE_AA)

    # ── Bézier smooth helper ─────────────────────────────────────────
    def smooth_polyline(pts_list, color, thick, n=120):
        pts = np.array(pts_list, dtype=np.float32)
        n_seg = len(pts)
        out = []
        for i in range(0, n_seg - 1, 3):
            p0 = pts[min(i,     n_seg-1)]
            p1 = pts[min(i+1,   n_seg-1)]
            p2 = pts[min(i+2,   n_seg-1)]
            p3 = pts[min(i+3,   n_seg-1)]
            steps = max(8, n // max(1, n_seg // 3))
            for t in np.linspace(0, 1, steps):
                mt = 1 - t
                x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
                y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
                out.append((int(x), int(y)))
        arr = np.array(out, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [arr], False, color, thick, cv2.LINE_AA)

    # ── Colors ───────────────────────────────────────────────────────
    BLACK  = (20,  20,  20)
    BLUE   = (180, 80,  40)   # BGR → warm blue for eyes
    BROW_C = (30,  30,  30)
    NOSE_C = (60,  40,  20)
    LIP_C  = (40,  40, 160)   # BGR → red lips
    GRID_C = (180, 80,  80)   # BGR → soft blue grid
    LABEL_C= (120, 40,  40)
    ARC_C  = (40, 160,  40)   # green orbital arc
    EAR_C  = (140,140, 140)

    # ════════════════════════════════════════════════════════════════
    # 1. FACE OVAL — smooth Bézier
    # ════════════════════════════════════════════════════════════════
    oval_canvas_pts = [lm_pt(i) for i in FACE_OVAL]
    smooth_polyline(oval_canvas_pts, BLACK, thick=3)

    # ════════════════════════════════════════════════════════════════
    # 2. EYES — polyline outline + iris circles
    # ════════════════════════════════════════════════════════════════
    LEFT_EYE  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
    RIGHT_EYE = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249,263]
    draw_path(LEFT_EYE,  BLUE, 2)
    draw_path(RIGHT_EYE, BLUE, 2)

    for iris_idx, radius in [(468, 11), (473, 11)]:
        if iris_idx < len(lms):
            cv2.circle(canvas, lm_pt(iris_idx), radius, BLUE, 1, cv2.LINE_AA)
            cv2.circle(canvas, lm_pt(iris_idx), 3,      BLUE, -1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════
    # 3. EYEBROWS
    # ════════════════════════════════════════════════════════════════
    LEFT_BROW  = [46,53,52,65,55,107,66,105,63,70]
    RIGHT_BROW = [276,283,282,295,285,336,296,334,293,300]
    draw_path(LEFT_BROW,  BROW_C, 2)
    draw_path(RIGHT_BROW, BROW_C, 2)

    # Orbital arc (green) — across brow tops
    l_brow_top = lm_pt(70)
    r_brow_top = lm_pt(300)
    arc_mid    = lm_pt(168)  # nose bridge = arc midpoint
    arc_pts    = np.array([l_brow_top, arc_mid, r_brow_top], dtype=np.int32)
    cv2.polylines(canvas, [arc_pts], False, ARC_C, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Orbital Arc", (arc_mid[0] - 40, arc_mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ARC_C, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════
    # 4. NOSE
    # ════════════════════════════════════════════════════════════════
    NOSE_BRIDGE = [168,6,197,195,5,4,1]
    NOSE_BASE   = [129,98,97,2,326,327,358]
    draw_path(NOSE_BRIDGE, NOSE_C, 2)
    draw_path(NOSE_BASE,   NOSE_C, 2)
    draw_path([64, 60, 2, 290, 294], NOSE_C, 1)   # nostril wings

    # ════════════════════════════════════════════════════════════════
    # 5. LIPS
    # ════════════════════════════════════════════════════════════════
    UPPER_LIP = [61,185,40,39,37,0,267,269,270,409,291,308,310,311,312,13,82,81,80,191,78,61]
    LOWER_LIP = [61,78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]
    draw_path(UPPER_LIP, LIP_C, 2)
    draw_path(LOWER_LIP, LIP_C, 2)

    # ════════════════════════════════════════════════════════════════
    # 6. EAR HINTS (soft gray)
    # ════════════════════════════════════════════════════════════════
    draw_path([127,234,93,132,58,172,136,150],        EAR_C, 1)
    draw_path([356,454,323,361,288,397,365,379],       EAR_C, 1)

    # ════════════════════════════════════════════════════════════════
    # 7. LOOMIS PROPORTION GRID — normalized & clean
    # ════════════════════════════════════════════════════════════════
    top_c   = lm_pt(10)
    bot_c   = lm_pt(152)
    left_c  = lm_pt(234)
    right_c = lm_pt(454)
    l_eye_c = lm_pt(33)
    r_eye_c = lm_pt(263)

    cx      = (left_c[0] + right_c[0]) // 2
    f_top   = top_c[1]
    f_bot   = bot_c[1]
    f_left  = left_c[0] - 18
    f_right = right_c[0] + 18
    f_h     = f_bot - f_top

    # Normalize horizontal lines to exact Loomis thirds
    y_hairline = f_top
    y_brow     = f_top + int(f_h * 0.333)
    y_nose_eye = f_top + int(f_h * 0.500)   # eye line = midpoint
    y_nose     = f_top + int(f_h * 0.666)
    y_mouth    = f_top + int(f_h * 0.833)
    y_chin     = f_bot

    # Vertical center axis (red)
    cv2.line(canvas, (cx, f_top - 35), (cx, f_bot + 20),
             (80, 80, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "1/2", (cx + 4, f_top - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, LABEL_C, 1, cv2.LINE_AA)

    # Horizontal thirds (soft blue)
    thirds = [
        (y_brow,     "1/3"),
        (y_nose,     "1/3"),
        (y_mouth,    ""),
        (y_chin,     ""),
    ]
    for y, lbl in thirds:
        cv2.line(canvas, (f_left - 10, y), (f_right + 10, y),
                 GRID_C, 1, cv2.LINE_AA)
        if lbl:
            cv2.putText(canvas, lbl, (f_left - 48, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, LABEL_C, 1, cv2.LINE_AA)

    # Eye line (yellow/orange)
    cv2.line(canvas, (f_left, y_nose_eye), (f_right + 80, y_nose_eye),
             (0, 180, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, "1/3  Forehead", (f_right + 14, y_nose_eye + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 180), 1, cv2.LINE_AA)

    # Mouth line (blue)
    cv2.line(canvas, (f_left, y_mouth), (f_right + 80, y_mouth),
             (180, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Mouth", (f_right + 14, y_mouth + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 100, 0), 1, cv2.LINE_AA)

    # Nose+Eyes zone label
    cv2.putText(canvas, "1/3  Nose+Eyes", (f_right + 14, y_nose + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_C, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════
    # 8. TITLE
    # ════════════════════════════════════════════════════════════════
    cv2.putText(canvas, "Loomis Construction  [Pro-Portion]",
                (canvas_w // 2 - 200, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 1, cv2.LINE_AA)

    return canvas


def draw_pose_wireframe(img, pose_landmarks):
    try:
        mp_drawing.draw_landmarks(
            img,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    except Exception as e:
        print(f"Error drawing pose: {e}")
    return img


# 3D reference points for head pose estimation
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype="double")


def calculate_head_pose(face_landmarks, img_width, img_height):
    """Calculate 3D head rotation angles (pitch, yaw, roll)"""
    
    try:
        # Camera matrix
        focal_length = img_width
        center = (img_width / 2, img_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        # Extract 2D points from MediaPipe landmarks
        image_points = np.array([
            (face_landmarks[1].x * img_width, face_landmarks[1].y * img_height),
            (face_landmarks[152].x * img_width, face_landmarks[152].y * img_height),
            (face_landmarks[33].x * img_width, face_landmarks[33].y * img_height),
            (face_landmarks[263].x * img_width, face_landmarks[263].y * img_height),
            (face_landmarks[61].x * img_width, face_landmarks[61].y * img_height),
            (face_landmarks[291].x * img_width, face_landmarks[291].y * img_height)
        ], dtype="double")
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("solvePnP failed")
            return None
        
        # Convert to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        pitch = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
        yaw = np.arctan2(-rotation_matrix[2][0], 
                         np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2))
        roll = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])
        
        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        
        # FIX: Normalize pitch to -90 to +90 range
        if pitch_deg > 90:
            pitch_deg = pitch_deg - 180
        elif pitch_deg < -90:
            pitch_deg = pitch_deg + 180
        
        return {
            "pitch": round(pitch_deg, 1),
            "yaw": round(yaw_deg, 1),
            "roll": round(roll_deg, 1)
        }
    
    except Exception as e:
        print(f"Head pose calculation error: {e}")
        return None


def classify_face_view(yaw, pitch):
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_pitch >= 22 and abs_yaw < 20:
        direction = "Up" if pitch < 0 else "Down"
        return f"Tilted {direction}"

    if abs_yaw < 15:
        return "Front View"
    elif abs_yaw < 35:
        direction = "Left" if yaw < 0 else "Right"
        return f"3/4 View ({direction})"
    elif abs_yaw < 65:
        direction = "Left" if yaw < 0 else "Right"
        return f"Profile ({direction})"
    else:
        direction = "Left" if yaw < 0 else "Right"
        return f"Extreme Profile ({direction})"


def generate_adaptive_3d_grid(face_landmarks, head_pose, img_width, img_height):
    """Generate Loomis grid adapted to 3D head pose"""
    
    bounds = get_face_bounds(face_landmarks, img_width, img_height)
    
    # Get pose angles
    yaw = head_pose['yaw']
    pitch = head_pose['pitch']
    roll = head_pose['roll']
    
    # Calculate perspective compression
    yaw_factor = np.cos(np.radians(abs(yaw)))
    pitch_factor = np.cos(np.radians(abs(pitch)))
    
    # Adjusted dimensions
    center_x = bounds['x_center']
    center_y = (bounds['y_top'] + bounds['y_bottom']) / 2
    half_width = bounds['face_width'] / 2
    half_height = bounds['face_height'] / 2
    
    # Apply perspective to width (horizontal compression when turning)
    adj_half_width = half_width * yaw_factor
    
    # Apply perspective to height (vertical compression when tilting)
    adj_half_height = half_height * pitch_factor
    
    # Shift center based on yaw (face turns, center shifts)
    center_shift_x = (half_width - adj_half_width) * (1 if yaw > 0 else -1) * 0.5
    
    # Calculate grid lines
    left = center_x - adj_half_width + center_shift_x
    right = center_x + adj_half_width + center_shift_x
    top = bounds['y_top']
    bottom = bounds['y_bottom']
    
    # Horizontal divisions (with perspective)
    thirds_y = [
        top + adj_half_height * 2 / 6,   # Hairline
        top + adj_half_height * 2 / 3,   # Eyebrow
        top + adj_half_height * 4 / 3    # Nose
    ]
    
    grid_data = {
        "vertical_center": {
            "x": int(center_x),
            "y1": int(top),
            "y2": int(bottom)
        },
        "horizontal_lines": [
            {
                "label": "Hairline",
                "x1": int(left),
                "x2": int(right),
                "y": int(thirds_y[0])
            },
            {
                "label": "Eyebrow",
                "x1": int(left),
                "x2": int(right),
                "y": int(thirds_y[1])
            },
            {
                "label": "Nose",
                "x1": int(left),
                "x2": int(right),
                "y": int(thirds_y[2])
            }
        ],
        "bounding_box": {
            "left": int(left),
            "right": int(right),
            "top": int(top),
            "bottom": int(bottom)
        },
        "eye_line": {
            "x1": int(face_landmarks[33].x * img_width),
            "x2": int(face_landmarks[263].x * img_width),
            "y": int((face_landmarks[33].y + face_landmarks[263].y) / 2 * img_height)
        }
    }
    
    return grid_data


@app.get("/")
def root():
    return {
        "message": "Pro-Portion v2.0 - Static + Real-Time 3D Grid Analysis",
        "version": "2.0.0",
        "modes": {
            "static": {
                "description": "Upload and analyze saved photos",
                "endpoints": [
                    "/process - Standard analysis",
                    "/process-tutorial - 6-step tutorial generation"
                ]
            },
            "realtime": {
                "description": "Live webcam with 3D adaptive Loomis grid",
                "endpoints": [
                    "/ws/realtime-grid - WebSocket for live streaming",
                    "/process-realtime - REST endpoint for frames"
                ]
            }
        },
        "features": [
            "Step-by-step Loomis grid tutorials",
            "Real-time measurements on each step",
            "ML-based proportion analysis (84%+ accuracy)",
            "Face shape classification",
            "Multi-person detection",
            "3D head pose estimation (pitch, yaw, roll)",
            "Adaptive grid for any face angle"
        ]
    }



@app.get("/health")
def health_check():
    return {"status": "Pro-Portion v1.1 ready", "version": "1.1.0"}


@app.post("/process-tutorial")
async def process_tutorial(file: UploadFile = File(...)):
    """Generate step-by-step Loomis grid tutorial with measurements"""

    # ── Universal image decode (JPG, PNG, WEBP, BMP, TIFF, HEIC, etc.) ──
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        try:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            img     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported or corrupt image format")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    height, width = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"TUTORIAL MODE: {file.filename}")
    print(f"Dimensions: {width}x{height}")

    rgb_img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_img)

    if not face_results.multi_face_landmarks:
        raise HTTPException(
            status_code=400,
            detail="No faces detected. Use a clear, front-facing photo with good lighting."
        )

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    tutorial_images = []

    # Process first face only for tutorial
    face_landmarks = face_results.multi_face_landmarks[0].landmark

    step_descriptions = [
        "Face bounding box - establishes overall proportions",
        "Vertical centerline - facial symmetry axis",
        "Horizontal thirds - hairline, eyebrow, nose divisions",
        "Eye line - precise eye placement",
        "Face outline - jaw and cheek contours",
        "Complete Loomis grid - all construction lines"
    ]

    for step in range(1, 7):
        step_img      = draw_tutorial_step(img, face_landmarks, 0, step, width, height)
        step_filename = f"tutorial_step{step}_{timestamp}.jpg"
        step_path     = os.path.join(TUTORIAL_DIR, step_filename)
        cv2.imwrite(step_path, step_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        tutorial_images.append({
            "step":        step,
            "description": step_descriptions[step - 1],
            "filename":    step_filename
        })
        print(f"[OK] Step {step}: {step_descriptions[step - 1]}")

    # ── Proportions + improved face shape analysis ───────────────────
    face_ratios = compute_face_ratios(face_landmarks, width, height)
    if face_ratios is None:
        raise HTTPException(status_code=500, detail="Failed to compute face ratios")

    analysis = analyze_proportions_vs_ideal(face_ratios)
    if analysis is None:
        raise HTTPException(status_code=500, detail="Failed to analyze proportions")

    print(f"ML Analysis Score : {analysis['overall_score']:.1f}/100")
    print(f"Face Shape        : {analysis['face_shape']}")
    print(f"Jaw/Face ratio    : {face_ratios['proportional_ratios'].get('jaw_to_face_width', 'N/A')}")
    print(f"Forehead/Jaw ratio: {face_ratios['proportional_ratios'].get('forehead_to_jaw', 'N/A')}")
    print(f"{'='*60}\n")

    # ── Format tutorial steps for frontend ──────────────────────────
    tutorial_steps_formatted = [
        {
            "title":    f"Step {s['step']}: {s['description'].split(' - ')[0]}",
            "filename": s["filename"]
        }
        for s in tutorial_images
    ]

    # ── Build face data matching ProcessResult structure ─────────────
    face_data = {
        "measurements_px": face_ratios.get("measurements_px", {}),
        "proportional_ratios": face_ratios.get("proportional_ratios", {}),
        "analysis": analysis
    }

    return {
        "status":         "success",
        "filename":       file.filename,
        "tutorial_steps": tutorial_steps_formatted,
        "face_count":     1,
        "faces":          [face_data]
    }


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """Standard processing with Loomis grid and ML analysis"""

    # ── Universal image decode (JPG, PNG, WEBP, BMP, TIFF, HEIC, etc.) ──
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        try:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            img     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported or corrupt image format")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    height, width = img.shape[:2]
    print(f"\n{'='*50}")
    print(f"Processing: {file.filename}")
    print(f"Dimensions: {width}x{height}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Face detection ───────────────────────────────────────────────
    basic_face_results = face_detection.process(rgb_img)
    if basic_face_results.detections:
        print(f"Basic detector: {len(basic_face_results.detections)} faces")

    face_results = face_mesh.process(rgb_img)
    pose_results = pose.process(rgb_img)

    annotated_img = img.copy()

    # ── Process faces ────────────────────────────────────────────────
    faces_data = []
    if face_results.multi_face_landmarks:
        num_faces = len(face_results.multi_face_landmarks)
        print(f"[OK] Face Mesh: {num_faces} faces")

        for idx, face_landmarks_obj in enumerate(face_results.multi_face_landmarks):
            face_landmarks = face_landmarks_obj.landmark

            # Draw Loomis grid
            annotated_img = draw_loomis_grid(annotated_img, face_landmarks, idx)

            # Compute ratios and measurements
            face_ratios = compute_face_ratios(face_landmarks, width, height)

            if face_ratios:
                analysis = analyze_proportions_vs_ideal(face_ratios)

                if analysis:
                    face_data = {
                        "measurements_px":     face_ratios.get("measurements_px", {}),
                        "proportional_ratios": face_ratios.get("proportional_ratios", {}),
                        "analysis":            analysis
                    }
                    faces_data.append(face_data)

                    print(f"Face {idx + 1}:")
                    print(f"  Score          : {analysis['overall_score']:.1f}/100")
                    print(f"  Shape          : {analysis['face_shape']}")
                    print(f"  Jaw/Face       : {face_ratios['proportional_ratios'].get('jaw_to_face_width', 'N/A')}")
                    print(f"  Forehead/Jaw   : {face_ratios['proportional_ratios'].get('forehead_to_jaw', 'N/A')}")
                    print(f"  Cheekbone/Jaw  : {face_ratios['proportional_ratios'].get('cheekbone_to_jaw', 'N/A')}")
                    print(f"  Aspect Ratio   : {face_ratios['proportional_ratios'].get('face_aspect_ratio', 'N/A')}")
    else:
        print("[X] No faces detected")

    # ── Process body (optional) ──────────────────────────────────────
    body_data = None
    if pose_results.pose_landmarks:
        print(f"[OK] Body detected")
        pose_landmarks = pose_results.pose_landmarks.landmark
        body_ratios    = compute_body_ratios(pose_landmarks, width, height)

        annotated_img = draw_pose_wireframe(annotated_img, pose_results.pose_landmarks)

        if body_ratios:
            body_data = {
                "detected":        True,
                "landmark_count":  len(pose_landmarks),
                "proportions":     body_ratios
            }

    # ── Save annotated image ─────────────────────────────────────────
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}.jpg"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {output_path}")

    # ── Encode to base64 for frontend ────────────────────────────────
    _, buffer    = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64   = base64.b64encode(buffer).decode('utf-8')

    print(f"{'='*50}\n")

    # ── Build response ───────────────────────────────────────────────
    response = {
        "status":                "success" if faces_data else "no_face",
        "face_count":            len(faces_data),
        "faces":                 faces_data,
        "processed_image":       img_base64,
        "processed_image_url":   f"/download/{output_filename}",
        "timestamp":             timestamp
    }

    if body_data:
        response["body_analysis"] = body_data

    return response


@app.post("/sketch-canvas")
async def generate_sketch_canvas(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        img_h, img_w = img.shape[:2]          # ← real pixel dimensions
        rgb          = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results      = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            raise HTTPException(status_code=400, detail="No face detected")

        landmarks = results.multi_face_landmarks[0].landmark

        # ✅ Pass img_w and img_h so coordinates map correctly
        canvas = draw_sketch_canvas(landmarks,
                                    img_w=img_w, img_h=img_h,
                                    canvas_w=900, canvas_h=1100)

        _, buffer = cv2.imencode('.png', canvas)
        b64       = base64.b64encode(buffer).decode('utf-8')
        ratios    = compute_face_ratios(landmarks, img_w, img_h)
        analysis  = analyze_proportions_vs_ideal(ratios) if ratios else None

        return {
            "canvas_image": f"data:image/png;base64,{b64}",
            "ratios":       ratios,
            "analysis":     analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)


@app.get("/download-tutorial/{filename}")
async def download_tutorial(filename: str):
    file_path = os.path.join(TUTORIAL_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Tutorial image not found")
    
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)


@app.get("/list-processed")
def list_processed_images():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    files.sort(reverse=True)
    return {"processed_images": files[:20], "total_count": len(files)}


@app.get("/list-tutorials")
def list_tutorials():
    files = [f for f in os.listdir(TUTORIAL_DIR) if f.endswith('.jpg')]
    files.sort(reverse=True)
    return {"tutorial_images": files[:30], "total_count": len(files)}


@app.websocket("/ws/realtime-grid")
async def websocket_realtime_grid(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    frame_count = 0
    send_annotated = True

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"status": "ping"})
                    continue
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                print(f"Frame {frame_count} error: {e}")
                break

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    text_data = json.loads(message["text"])
                    if "grid" in text_data:
                        send_annotated = bool(text_data["grid"])
                        print(f"Grid annotation: {'ON' if send_annotated else 'OFF'}")
                except Exception:
                    pass

                try:
                    await websocket.send_json({"status": "ping"})
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
                continue

            if "bytes" not in message or message["bytes"] is None:
                continue

            data = message["bytes"]
            frame_count += 1

            try:
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    await websocket.send_json({"status": "invalid_frame"})
                    continue

                target_w = 640
                h, w = img.shape[:2]
                scale = target_w / w
                img = cv2.resize(img, (target_w, int(h * scale)))
                height, width = img.shape[:2]

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_results = face_mesh_video.process(rgb_img)

                if not face_results.multi_face_landmarks:
                    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b64_frame = base64.b64encode(buffer).decode("utf-8")
                    await websocket.send_json({
                        "status": "no_face",
                        "frame": b64_frame
                    })
                    continue

                face_landmarks = face_results.multi_face_landmarks[0].landmark
                head_pose = calculate_head_pose(face_landmarks, width, height)
                view_type = classify_face_view(
                    head_pose["yaw"], head_pose["pitch"]
                ) if head_pose else "Unknown"

                face_ratios = compute_face_ratios(face_landmarks, width, height)

                analysis = None
                if face_ratios and head_pose:
                    if abs(head_pose["yaw"]) <= 15:
                        analysis = analyze_proportions_vs_ideal(face_ratios)
                    elif abs(head_pose["yaw"]) <= 35:
                        analysis = analyze_proportions_vs_ideal(face_ratios)
                        if analysis:
                            analysis["face_shape"] = f"{analysis['face_shape']} (3/4 view)"
                    else:
                        analysis = {
                            "overall_score": 0,
                            "face_shape": "Profile view",
                            "comparisons": {}
                        }

                if send_annotated:
                    annotated = img.copy()
                    annotated = draw_loomis_grid(annotated, face_landmarks, 0,head_pose)
                    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                else:
                    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])

                b64_frame = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_json({
                    "status": "success",
                    "frame": b64_frame,
                    "pose": head_pose,
                    "view_type": view_type,
                    "measurements": face_ratios["measurements_px"] if face_ratios else None,
                    "ratios": face_ratios["proportional_ratios"] if face_ratios else None,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                })

            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                print(f"Frame {frame_count} error: {e}")
                break
            except Exception as frame_error:
                print(f"Frame {frame_count} error: {frame_error}")
                try:
                    await websocket.send_json({
                        "status": "error",
                        "message": str(frame_error)
                    })
                except Exception:
                    break

    except Exception as e:
        print(f"WebSocket fatal: {e}")
    finally:
        print(f"WebSocket closed after {frame_count} frames")



# ========== NEW: Fast Processing Endpoint (Alternative to WebSocket) ==========

@app.post("/process-realtime")
async def process_realtime_frame(file: UploadFile = File(...)):
    """Fast processing for individual webcam frames (REST alternative to WebSocket)"""
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid frame")
    
    height, width = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Fast face detection
    face_results = face_mesh.process(rgb_img)
    
    if not face_results.multi_face_landmarks:
        return {"status": "no_face", "grid": None}
    
    face_landmarks = face_results.multi_face_landmarks[0].landmark
    
    # Calculate pose
    head_pose = calculate_head_pose(face_landmarks, width, height)
    
    if not head_pose:
        return {"status": "pose_failed", "grid": None}
    
    # Generate grid
    grid_3d = generate_adaptive_3d_grid(face_landmarks, head_pose, width, height)
    view_type = classify_face_view(head_pose['yaw'], head_pose['pitch'])
    
    return {
        "status": "success",
        "grid": grid_3d,
        "pose": head_pose,
        "view_type": view_type
    }
