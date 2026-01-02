from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import mediapipe as mp
import math
import base64

app = FastAPI(title="Pro-Portion Backend v0.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

def calculate_distance(landmark1, landmark2, img_width, img_height):
    x1, y1 = landmark1.x * img_width, landmark1.y * img_height
    x2, y2 = landmark2.x * img_width, landmark2.y * img_height
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def compute_face_ratios(landmarks, img_width, img_height):
    eye_distance = calculate_distance(landmarks[33], landmarks[362], img_width, img_height)
    nose_to_chin = calculate_distance(landmarks[1], landmarks[152], img_width, img_height)
    face_width = calculate_distance(landmarks[234], landmarks[454], img_width, img_height)
    forehead_to_chin = calculate_distance(landmarks[10], landmarks[152], img_width, img_height)
    
    return {
        "measurements_px": {
            "eye_distance": round(eye_distance, 2),
            "nose_to_chin": round(nose_to_chin, 2),
            "face_width": round(face_width, 2),
            "face_height": round(forehead_to_chin, 2)
        },
        "proportional_ratios": {
            "eye_to_face_width": round(eye_distance / face_width, 3) if face_width > 0 else 0,
            "nose_to_face_height": round(nose_to_chin / forehead_to_chin, 3) if forehead_to_chin > 0 else 0,
            "face_aspect_ratio": round(face_width / forehead_to_chin, 3) if forehead_to_chin > 0 else 0
        }
    }

def compute_body_ratios(landmarks, img_width, img_height):
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

# Draw Loomis Method grid on face
def draw_loomis_grid(img, face_landmarks):
    height, width = img.shape[:2]
    
    # Key facial landmarks for Loomis grid
    top = face_landmarks[10]  # Forehead
    bottom = face_landmarks[152]  # Chin
    left = face_landmarks[234]  # Left edge
    right = face_landmarks[454]  # Right edge
    nose_tip = face_landmarks[1]
    
    # Convert to pixel coordinates
    x_center = int((left.x + right.x) / 2 * width)
    y_top = int(top.y * height)
    y_bottom = int(bottom.y * height)
    x_left = int(left.x * width)
    x_right = int(right.x * width)
    y_nose = int(nose_tip.y * height)
    
    # Draw vertical center line
    cv2.line(img, (x_center, y_top), (x_center, y_bottom), (0, 255, 0), 2)
    
    # Draw horizontal lines (thirds)
    face_height = y_bottom - y_top
    y_third1 = y_top + face_height // 3
    y_third2 = y_top + 2 * face_height // 3
    
    cv2.line(img, (x_left, y_third1), (x_right, y_third1), (0, 255, 0), 2)
    cv2.line(img, (x_left, y_third2), (x_right, y_third2), (0, 255, 0), 2)
    
    # Draw face bounding box
    cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)
    
    # Draw eye line
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[362]
    cv2.line(img, 
             (int(left_eye.x * width), int(left_eye.y * height)),
             (int(right_eye.x * width), int(right_eye.y * height)),
             (255, 0, 0), 2)
    
    # Add labels
    cv2.putText(img, "Loomis Grid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img

# Draw skeletal wireframe on body
def draw_pose_wireframe(img, pose_landmarks):
    # Use MediaPipe's built-in drawing utilities for pose skeleton
    mp_drawing.draw_landmarks(
        img,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    # Add label
    cv2.putText(img, "Pose Skeleton", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img

@app.get("/health")
def health_check():
    return {"status": "Pro-Portion v0.6 - Overlays Ready", "version": "0.6.0"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images allowed")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    height, width = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process detections
    face_results = face_mesh.process(rgb_img)
    pose_results = pose.process(rgb_img)
    
    # Create annotated image copy
    annotated_img = img.copy()
    
    response = {
        "status": "Processing complete",
        "filename": file.filename,
        "image_dimensions": {"width": width, "height": height},
        "face_analysis": {"detected": False},
        "body_analysis": {"detected": False}
    }
    
    # Process face
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        face_ratios = compute_face_ratios(face_landmarks, width, height)
        
        # Draw Loomis grid
        annotated_img = draw_loomis_grid(annotated_img, face_landmarks)
        
        response["face_analysis"] = {
            "detected": True,
            "landmark_count": len(face_landmarks),
            "proportions": face_ratios,
            "overlay": "loomis_grid_applied"
        }
    
    # Process body
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark
        body_ratios = compute_body_ratios(pose_landmarks, width, height)
        
        # Draw skeletal wireframe
        annotated_img = draw_pose_wireframe(annotated_img, pose_results.pose_landmarks)
        
        response["body_analysis"] = {
            "detected": True,
            "landmark_count": len(pose_landmarks),
            "proportions": body_ratios,
            "overlay": "skeleton_applied"
        }
    
    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    response["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
    
    return response
