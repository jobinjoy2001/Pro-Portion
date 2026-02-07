import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

// ============================================
// TYPE DEFINITIONS
// ============================================

export interface ProcessResult {
  status: string;
  filename: string;
  faces_detected: number;
  processed_image: string;
  measurements: {
    face_width: number;
    face_height: number;
    eye_distance: number;
    nose_to_chin: number;
  };
  proportion_score: number;
  face_shape: string;
  symmetry?: number;
  ml_analyses: Array<{
    face_width: number;
    face_height: number;
    eye_distance: number;
    nose_chin_ratio: number;
    thirds: { upper: number; middle: number; lower: number };
  }>;
}

export interface TutorialStep {
  title: string;
  filename: string;
}

export interface TutorialResult {
  status: string;
  filename: string;
  tutorial_steps: TutorialStep[];
  proportion_score: number;
  face_shape: string;
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
    console.log('✅ WebSocket connected to backend');
  };
  
  ws.onerror = (error) => {
    console.error('❌ WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected');
  };
  
  return ws;
}

export default api;
