import axios from 'axios';
import { API_ENDPOINTS } from '@/config/api';

// ============================================
// STATIC IMAGE ANALYSIS
// ============================================

export interface ProcessImageResponse {
  status: string;
  filename: string;
  faces_detected: number;
  processed_image: string;
  measurements?: {
    face_width: number;
    face_height: number;
    eye_distance: number;
    nose_to_chin: number;
  };
  proportion_score?: number;
  face_shape?: string;
  symmetry?: number;
  ml_analyses?: any;
}

export async function uploadImageForAnalysis(
  file: File
): Promise<ProcessImageResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post<ProcessImageResponse>(
    API_ENDPOINTS.PROCESS_IMAGE,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
    }
  );

  return response.data;
}

// ============================================
// TUTORIAL MODE
// ============================================

export interface TutorialStep {
  title: string;
  filename: string;
}

export interface ProcessTutorialResponse {
  status: string;
  filename: string;
  tutorial_steps: TutorialStep[];
  proportion_score: number;
  face_shape: string;
}

export async function uploadImageForTutorial(
  file: File
): Promise<ProcessTutorialResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post<ProcessTutorialResponse>(
    API_ENDPOINTS.PROCESS_TUTORIAL,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
    }
  );

  return response.data;
}

// ============================================
// IMAGE DOWNLOAD HELPERS
// ============================================

export function getProcessedImageUrl(filename: string): string {
  return API_ENDPOINTS.DOWNLOAD_IMAGE(filename);
}

export function getTutorialImageUrl(filename: string): string {
  return API_ENDPOINTS.DOWNLOAD_TUTORIAL(filename);
}

// ============================================
// REAL-TIME WEBSOCKET
// ============================================

export interface RealtimeGridData {
  status: string;
  grid?: {
    vertical_center: { x: number; y1: number; y2: number };
    horizontal_lines: Array<{ label: string; x1: number; x2: number; y: number }>;
    bounding_box: { left: number; right: number; top: number; bottom: number };
  };
  pose?: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  view_type?: string;
}

export class RealtimeWebSocket {
  private ws: WebSocket | null = null;
  private onMessageCallback: ((data: RealtimeGridData) => void) | null = null;

  connect(onMessage: (data: RealtimeGridData) => void) {
    this.onMessageCallback = onMessage;
    this.ws = new WebSocket(API_ENDPOINTS.REALTIME_WEBSOCKET);

    this.ws.onopen = () => {
      console.log('✅ WebSocket connected to backend');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (this.onMessageCallback) {
          this.onMessageCallback(data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
  }

  sendFrame(imageBlob: Blob) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(imageBlob);
    } else {
      console.warn('WebSocket not ready. Current state:', this.ws?.readyState);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      console.log('WebSocket manually disconnected');
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
