// Backend server URLs
export const API_BASE_URL = 'http://localhost:8000';
export const WS_BASE_URL = 'ws://localhost:8000';

// API Endpoints
export const API_ENDPOINTS = {
  // Static image analysis
  PROCESS_IMAGE: `${API_BASE_URL}/process`,
  
  // Tutorial mode (6 steps)
  PROCESS_TUTORIAL: `${API_BASE_URL}/process-tutorial`,
  
  // Download processed images
  DOWNLOAD_IMAGE: (filename: string) => `${API_BASE_URL}/download/${filename}`,
  
  // Download tutorial steps
  DOWNLOAD_TUTORIAL: (filename: string) => `${API_BASE_URL}/download-tutorial/${filename}`,
  
  // WebSocket for real-time camera
  REALTIME_WEBSOCKET: `${WS_BASE_URL}/ws/realtime-grid`,
};
