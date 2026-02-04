from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import mediapipe as mp
import math
import os
from datetime import datetime
from sklearn.svm import SVR
import joblib


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


face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.2
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
        eye_distance = calculate_distance(landmarks[33], landmarks[362], img_width, img_height)
        nose_to_chin = calculate_distance(landmarks[1], landmarks[152], img_width, img_height)
        face_width = calculate_distance(landmarks[234], landmarks[454], img_width, img_height)
        forehead_to_chin = calculate_distance(landmarks[10], landmarks[152], img_width, img_height)
        
        # Additional measurements
        mouth_width = calculate_distance(landmarks[61], landmarks[291], img_width, img_height)
        nose_width = calculate_distance(landmarks[48], landmarks[278], img_width, img_height)
        
        return {
            "measurements_px": {
                "eye_distance": round(eye_distance, 2),
                "nose_to_chin": round(nose_to_chin, 2),
                "face_width": round(face_width, 2),
                "face_height": round(forehead_to_chin, 2),
                "mouth_width": round(mouth_width, 2),
                "nose_width": round(nose_width, 2)
            },
            "proportional_ratios": {
                "eye_to_face_width": round(eye_distance / face_width, 3) if face_width > 0 else 0,
                "nose_to_face_height": round(nose_to_chin / forehead_to_chin, 3) if forehead_to_chin > 0 else 0,
                "face_aspect_ratio": round(face_width / forehead_to_chin, 3) if forehead_to_chin > 0 else 0,
                "mouth_to_face_width": round(mouth_width / face_width, 3) if face_width > 0 else 0,
                "nose_to_face_width": round(nose_width / face_width, 3) if face_width > 0 else 0
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
    
    analysis = {
        "overall_score": 0,
        "comparisons": {},
        "recommendations": []
    }
    
    # Compare each ratio
    scores = []
    
    # Eye spacing
    eye_diff = abs(detected["eye_to_face_width"] - IDEAL_PROPORTIONS["eye_to_face_width"])
    eye_score = max(0, 100 - (eye_diff * 200))  # Convert to 0-100 scale
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
    
    # Face shape classification
    aspect = detected["face_aspect_ratio"]
    if aspect > 0.85:
        analysis["face_shape"] = "Round/Square"
    elif aspect < 0.65:
        analysis["face_shape"] = "Oblong/Long"
    else:
        analysis["face_shape"] = "Oval/Balanced"
    
    if not analysis["recommendations"]:
        analysis["recommendations"].append("Proportions closely match classical ideal!")
    
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


def get_face_bounds(face_landmarks, width, height):
    """Extract face boundary coordinates"""
    top = face_landmarks[10]
    bottom = face_landmarks[152]
    left = face_landmarks[234]
    right = face_landmarks[454]
    
    return {
        'x_center': int((left.x + right.x) / 2 * width),
        'y_top': int(top.y * height),
        'y_bottom': int(bottom.y * height),
        'x_left': int(left.x * width),
        'x_right': int(right.x * width),
        'face_height': int(bottom.y * height) - int(top.y * height),
        'face_width': int(right.x * width) - int(left.x * width)
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


def draw_tutorial_step(img, face_landmarks, face_id, step_number, width, height):
    """Draw progressive Loomis grid construction steps"""
    bounds = get_face_bounds(face_landmarks, width, height)
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    color = colors[face_id % len(colors)]
    
    canvas = img.copy()
    
    # Calculate key lines
    face_height = bounds['y_bottom'] - bounds['y_top']
    y_hairline = bounds['y_top'] + face_height // 6
    y_eyebrow = bounds['y_top'] + face_height // 3
    y_nose = bounds['y_top'] + 2 * face_height // 3
    
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[362]
    y_eye_line = int((left_eye.y + right_eye.y) / 2 * height)
    
    step_name = ""
    
    if step_number == 1:
        # Step 1: Bounding box
        cv2.rectangle(canvas, 
                     (bounds['x_left'], bounds['y_top']), 
                     (bounds['x_right'], bounds['y_bottom']), 
                     color, 4)
        step_name = "Step 1: Face Bounds"
        cv2.putText(canvas, step_name, 
                   (bounds['x_left'], bounds['y_top'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    elif step_number == 2:
        # Step 2: Vertical centerline
        cv2.rectangle(canvas, 
                     (bounds['x_left'], bounds['y_top']), 
                     (bounds['x_right'], bounds['y_bottom']), 
                     color, 4)
        cv2.line(canvas, 
                (bounds['x_center'], bounds['y_top']), 
                (bounds['x_center'], bounds['y_bottom']), 
                color, 3)
        step_name = "Step 2: Center Line"
        cv2.putText(canvas, step_name, 
                   (bounds['x_left'], bounds['y_top'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    elif step_number == 3:
        # Step 3: Horizontal thirds
        cv2.rectangle(canvas, 
                     (bounds['x_left'], bounds['y_top']), 
                     (bounds['x_right'], bounds['y_bottom']), 
                     color, 4)
        cv2.line(canvas, 
                (bounds['x_center'], bounds['y_top']), 
                (bounds['x_center'], bounds['y_bottom']), 
                color, 3)
        
        # Hairline
        cv2.line(canvas, 
                (bounds['x_left'], y_hairline), 
                (bounds['x_right'], y_hairline), 
                (255, 200, 0), 2)
        cv2.putText(canvas, "Hairline", 
                   (bounds['x_right'] + 10, y_hairline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        # Eyebrow
        cv2.line(canvas, 
                (bounds['x_left'], y_eyebrow), 
                (bounds['x_right'], y_eyebrow), 
                (200, 150, 0), 2)
        cv2.putText(canvas, "Eyebrow", 
                   (bounds['x_right'] + 10, y_eyebrow),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 0), 2)
        
        # Nose
        cv2.line(canvas, 
                (bounds['x_left'], y_nose), 
                (bounds['x_right'], y_nose), 
                (150, 100, 0), 2)
        cv2.putText(canvas, "Nose", 
                   (bounds['x_right'] + 10, y_nose),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 100, 0), 2)
        
        step_name = "Step 3: Horizontal Thirds"
        cv2.putText(canvas, step_name, 
                   (bounds['x_left'], bounds['y_top'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    elif step_number == 4:
        # Step 4: Eye line
        cv2.rectangle(canvas, 
                     (bounds['x_left'], bounds['y_top']), 
                     (bounds['x_right'], bounds['y_bottom']), 
                     color, 4)
        cv2.line(canvas, 
                (bounds['x_center'], bounds['y_top']), 
                (bounds['x_center'], bounds['y_bottom']), 
                color, 3)
        cv2.line(canvas, (bounds['x_left'], y_hairline), (bounds['x_right'], y_hairline), (255, 200, 0), 2)
        cv2.line(canvas, (bounds['x_left'], y_eyebrow), (bounds['x_right'], y_eyebrow), (200, 150, 0), 2)
        cv2.line(canvas, (bounds['x_left'], y_nose), (bounds['x_right'], y_nose), (150, 100, 0), 2)
        
        # Eye line
        cv2.line(canvas, 
                (int(left_eye.x * width), y_eye_line), 
                (int(right_eye.x * width), y_eye_line), 
                (0, 255, 255), 3)
        cv2.putText(canvas, "Eye Line", 
                   (bounds['x_right'] + 10, y_eye_line),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        step_name = "Step 4: Eye Line"
        cv2.putText(canvas, step_name, 
                   (bounds['x_left'], bounds['y_top'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    elif step_number == 5:
        # Step 5: Face outline
        cv2.rectangle(canvas, 
                     (bounds['x_left'], bounds['y_top']), 
                     (bounds['x_right'], bounds['y_bottom']), 
                     color, 4)
        cv2.line(canvas, (bounds['x_center'], bounds['y_top']), (bounds['x_center'], bounds['y_bottom']), color, 3)
        cv2.line(canvas, (bounds['x_left'], y_hairline), (bounds['x_right'], y_hairline), (255, 200, 0), 2)
        cv2.line(canvas, (bounds['x_left'], y_eyebrow), (bounds['x_right'], y_eyebrow), (200, 150, 0), 2)
        cv2.line(canvas, (bounds['x_left'], y_nose), (bounds['x_right'], y_nose), (150, 100, 0), 2)
        cv2.line(canvas, (int(left_eye.x * width), y_eye_line), (int(right_eye.x * width), y_eye_line), (0, 255, 255), 3)
        
        # Jaw contour
        jaw_points = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 454]
        contour_pts = []
        for idx in jaw_points:
            pt = face_landmarks[idx]
            contour_pts.append([int(pt.x * width), int(pt.y * height)])
        contour_pts = np.array(contour_pts, dtype=np.int32)
        cv2.polylines(canvas, [contour_pts], False, (255, 0, 255), 2)
        
        step_name = "Step 5: Face Outline"
        cv2.putText(canvas, step_name, 
                   (bounds['x_left'], bounds['y_top'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    elif step_number == 6:
        # Step 6: Complete grid
        cv2.rectangle(canvas, 
                     (bounds['x_left'], bounds['y_top']), 
                     (bounds['x_right'], bounds['y_bottom']), 
                     color, 4)
        cv2.line(canvas, (bounds['x_center'], bounds['y_top']), (bounds['x_center'], bounds['y_bottom']), color, 3)
        cv2.line(canvas, (bounds['x_left'], y_hairline), (bounds['x_right'], y_hairline), (255, 200, 0), 2)
        cv2.line(canvas, (bounds['x_left'], y_eyebrow), (bounds['x_right'], y_eyebrow), (200, 150, 0), 2)
        cv2.line(canvas, (bounds['x_left'], y_nose), (bounds['x_right'], y_nose), (150, 100, 0), 2)
        cv2.line(canvas, (int(left_eye.x * width), y_eye_line), (int(right_eye.x * width), y_eye_line), (0, 255, 255), 3)
        
        # Nose marker
        nose_tip = face_landmarks[1]
        cv2.circle(canvas, (int(nose_tip.x * width), int(nose_tip.y * height)), 5, (0, 150, 255), -1)
        
        # Chin marker
        chin = face_landmarks[152]
        cv2.circle(canvas, (int(chin.x * width), int(chin.y * height)), 5, (255, 100, 0), -1)
        
        step_name = "Step 6: Complete Grid"
        cv2.putText(canvas, step_name, 
                   (bounds['x_left'], bounds['y_top'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    # Add measurements overlay to all steps
    canvas = add_measurements_overlay(canvas, face_landmarks, width, height, step_name)
    
    return canvas


def draw_loomis_grid(img, face_landmarks, face_id):
    """Draw complete Loomis grid (for /process endpoint)"""
    height, width = img.shape[:2]
    
    try:
        top = face_landmarks[10]
        bottom = face_landmarks[152]
        left = face_landmarks[234]
        right = face_landmarks[454]
        
        x_center = int((left.x + right.x) / 2 * width)
        y_top = int(top.y * height)
        y_bottom = int(bottom.y * height)
        x_left = int(left.x * width)
        x_right = int(right.x * width)
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                  (0, 255, 255), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)]
        color = colors[face_id % len(colors)]
        
        # Vertical center
        cv2.line(img, (x_center, y_top), (x_center, y_bottom), color, 3)
        
        # Horizontal thirds
        face_height = y_bottom - y_top
        if face_height > 0:
            y_third1 = y_top + face_height // 3
            y_third2 = y_top + 2 * face_height // 3
            cv2.line(img, (x_left, y_third1), (x_right, y_third1), color, 3)
            cv2.line(img, (x_left, y_third2), (x_right, y_third2), color, 3)
        
        # Bounding box
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), color, 4)
        
        # Eye line
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[362]
        cv2.line(img, 
                 (int(left_eye.x * width), int(left_eye.y * height)),
                 (int(right_eye.x * width), int(right_eye.y * height)),
                 color, 3)
        
        # Label
        label = f"Face {face_id + 1}"
        cv2.putText(img, label, (x_left, y_top - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
    except Exception as e:
        print(f"Error drawing grid for face {face_id + 1}: {e}")
    
    return img


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


@app.get("/")
def root():
    return {
        "message": "Pro-Portion v1.1 - Tutorial Mode + ML Analysis",
        "version": "1.1.0",
        "features": [
            "Step-by-step Loomis grid tutorials",
            "Real-time measurements on each step",
            "ML-based proportion analysis vs classical ideals",
            "Face shape classification",
            "Multi-person detection"
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "Pro-Portion v1.1 ready", "version": "1.1.0"}


@app.post("/process-tutorial")
async def process_tutorial(file: UploadFile = File(...)):
    """Generate step-by-step Loomis grid tutorial with measurements"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images allowed")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    height, width = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"TUTORIAL MODE: {file.filename}")
    print(f"Dimensions: {width}x{height}")
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_img)
    
    if not face_results.multi_face_landmarks:
        raise HTTPException(status_code=400, detail="No faces detected. Use clear, front-facing photo with good lighting.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tutorial_images = []
    
    # Process first face only for tutorial
    face_landmarks = face_results.multi_face_landmarks[0].landmark
    
    # Generate 6 tutorial steps
    step_descriptions = [
        "Face bounding box - establishes overall proportions",
        "Vertical centerline - facial symmetry axis",
        "Horizontal thirds - hairline, eyebrow, nose divisions",
        "Eye line - precise eye placement",
        "Face outline - jaw and cheek contours",
        "Complete Loomis grid - all construction lines"
    ]
    
    for step in range(1, 7):
        step_img = draw_tutorial_step(img, face_landmarks, 0, step, width, height)
        
        step_filename = f"tutorial_step{step}_{timestamp}.jpg"
        step_path = os.path.join(TUTORIAL_DIR, step_filename)
        cv2.imwrite(step_path, step_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        tutorial_images.append({
            "step": step,
            "description": step_descriptions[step - 1],
            "url": f"/download-tutorial/{step_filename}"
        })
        print(f"✓ Step {step}: {step_descriptions[step - 1]}")
    
    # Calculate proportions and ML analysis
    ratios = compute_face_ratios(face_landmarks, width, height)
    ml_analysis = analyze_proportions_vs_ideal(ratios)
    
    print(f"ML Analysis Score: {ml_analysis['overall_score']}/100")
    print(f"Face Shape: {ml_analysis['face_shape']}")
    print(f"{'='*60}\n")
    
    return {
        "status": "Tutorial generated successfully",
        "filename": file.filename,
        "image_dimensions": {"width": width, "height": height},
        "faces_detected": len(face_results.multi_face_landmarks),
        "tutorial_steps": tutorial_images,
        "measurements": ratios,
        "ml_analysis": ml_analysis
    }


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """Standard processing with Loomis grid and ML analysis"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images allowed")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    height, width = img.shape[:2]
    print(f"\n{'='*50}")
    print(f"Processing: {file.filename}")
    print(f"Dimensions: {width}x{height}")
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Face detection
    basic_face_results = face_detection.process(rgb_img)
    if basic_face_results.detections:
        print(f"Basic detector: {len(basic_face_results.detections)} faces")
    
    face_results = face_mesh.process(rgb_img)
    pose_results = pose.process(rgb_img)
    
    annotated_img = img.copy()
    
    response = {
        "status": "Processing complete",
        "filename": file.filename,
        "image_dimensions": {"width": width, "height": height},
        "faces_detected": 0,
        "bodies_detected": 0,
        "face_analyses": [],
        "body_analysis": None,
        "ml_analyses": []
    }
    
    # Process faces
    if face_results.multi_face_landmarks:
        num_faces = len(face_results.multi_face_landmarks)
        response["faces_detected"] = num_faces
        print(f"✓ Face Mesh: {num_faces} faces")
        
        for idx, face_landmarks_obj in enumerate(face_results.multi_face_landmarks):
            face_landmarks = face_landmarks_obj.landmark
            face_ratios = compute_face_ratios(face_landmarks, width, height)
            
            annotated_img = draw_loomis_grid(annotated_img, face_landmarks, idx)
            
            if face_ratios:
                ml_analysis = analyze_proportions_vs_ideal(face_ratios)
                
                response["face_analyses"].append({
                    "face_number": idx + 1,
                    "landmark_count": len(face_landmarks),
                    "proportions": face_ratios
                })
                
                response["ml_analyses"].append({
                    "face_number": idx + 1,
                    "analysis": ml_analysis
                })
    else:
        print("✗ No faces detected")
    
    # Process body
    if pose_results.pose_landmarks:
        print("✓ Body detected")
        pose_landmarks = pose_results.pose_landmarks.landmark
        body_ratios = compute_body_ratios(pose_landmarks, width, height)
        
        annotated_img = draw_pose_wireframe(annotated_img, pose_results.pose_landmarks)
        
        response["bodies_detected"] = 1
        if body_ratios:
            response["body_analysis"] = {
                "detected": True,
                "landmark_count": len(pose_landmarks),
                "proportions": body_ratios
            }
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {output_path}")
    print(f"{'='*50}\n")
    
    response["processed_image_url"] = f"/download/{output_filename}"
    
    return response


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
