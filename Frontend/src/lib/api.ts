import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

// ============================================
// TYPE DEFINITIONS
// ============================================

export interface FaceMeasurements {
  eye_distance?: number;
  nose_to_chin?: number;
  face_width?: number;
  face_height?: number;
  mouth_width?: number;
  nose_width?: number;
}

export interface ProportionalRatios {
  eye_to_face_width?: number;
  nose_to_face_height?: number;
  face_aspect_ratio?: number;
  mouth_to_face_width?: number;
  nose_to_face_width?: number;
}

export interface ProportionComparison {
  detected: number;
  ideal: number;
  score: number;
}

export interface FaceAnalysis {
  overall_score?: number;
  face_shape?: string;
  comparisons?: {
    [key: string]: ProportionComparison;
  };
  recommendations?: string[];
}

export interface FaceData {
  measurements_px?: FaceMeasurements;
  proportional_ratios?: ProportionalRatios;
  analysis?: FaceAnalysis;
}

export interface ProcessResult {
  status: string;
  face_count?: number;
  faces?: FaceData[];
  processed_image?: string;
  tutorial_steps?: string[];
  timestamp?: string;
}

export interface TutorialStep {
  title: string;
  filename: string;
}

export interface TutorialResult {
  status: string;
  filename: string;
  tutorial_steps: TutorialStep[];
  face_count?: number;
  faces?: FaceData[];
}

export interface RealtimeGridData {
  status: string;
  grid?: {
    vertical_center?: { 
      x: number; 
      y1: number; 
      y2: number 
    };
    horizontal_lines?: Array<{ 
      label: string; 
      x1: number; 
      x2: number; 
      y: number 
    }>;
    bounding_box?: { 
      left: number; 
      right: number; 
      top: number; 
      bottom: number 
    };
    eye_line?: { 
      x1: number; 
      x2: number; 
      y: number 
    };
  };
  pose?: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  view_type?: string;
  timestamp?: string;
}

// ============================================
// SKETCH CANVAS
// ============================================

export interface SketchCanvasResult {
  canvas_image: string;  // base64 data URL
  ratios: {
    measurements_px: {
      eye_distance: number;
      nose_to_chin: number;
      face_width: number;
      face_height: number;
      mouth_width: number;
      nose_width: number;
    };
    proportional_ratios: {
      eye_to_face_width: number;
      nose_to_face_height: number;
      face_aspect_ratio: number;
      mouth_to_face_width: number;
      nose_to_face_width: number;
    };
  };
  analysis: {
    overall_score: number;
    face_shape: string;
    recommendations: string[];
  };
}

// ============================================
// STATIC IMAGE ANALYSIS
// ============================================

export async function processImage(file: File): Promise<ProcessResult> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<ProcessResult>('/process', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  
  return response.data;
}

// ============================================
// TUTORIAL MODE
// ============================================

export async function processTutorial(file: File): Promise<TutorialResult> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<TutorialResult>('/process-tutorial', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  
  return response.data;
}

export async function generateSketchCanvas(file: File): Promise<SketchCanvasResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<SketchCanvasResult>('/sketch-canvas', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}


// ============================================
// DOWNLOAD HELPERS
// ============================================

export function getDownloadUrl(filename: string): string {
  return `${API_BASE_URL}/download/${filename}`;
}

export function getTutorialDownloadUrl(filename: string): string {
  return `${API_BASE_URL}/download-tutorial/${filename}`;
}

// ============================================
// REAL-TIME WEBSOCKET
// ============================================

export function createRealtimeWebSocket(): WebSocket {
  const ws = new WebSocket(`ws://localhost:8000/ws/realtime-grid`);
  
  ws.onopen = () => {
    console.log('WebSocket connected to backend');
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected');
  };
  
  return ws;
}

export default api;
