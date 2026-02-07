import { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import Webcam from "react-webcam";
import { Camera, CameraOff, Eye, EyeOff, AlertCircle } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import LiveInfoPanel from "@/components/LiveInfoPanel";
import { createRealtimeWebSocket, RealtimeGridData } from "@/lib/api";

const DrawMode = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animFrameRef = useRef<number>(0);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [gridData, setGridData] = useState<RealtimeGridData | null>(null);
  const [fps, setFps] = useState(0);
  const [permissionError, setPermissionError] = useState(false);

  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() });
  const frameCountRef = useRef(0);

  // Draw Loomis grid on canvas
  const drawGrid = useCallback((data: RealtimeGridData) => {
    if (!canvasRef.current || !webcamRef.current || !showGrid) return;

    const canvas = canvasRef.current;
    const video = webcamRef.current.video;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas to match video display size exactly
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Backend sends coordinates based on actual captured resolution
    // Use actual video dimensions for scaling
    const backendWidth = video.videoWidth;
    const backendHeight = video.videoHeight;
    
    const scaleX = canvas.width / backendWidth;
    const scaleY = canvas.height / backendHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // If grid data exists, draw it
    if (data.grid && data.status === "success") {
      const grid = data.grid;

      // Draw bounding box (GREEN)
      if (grid.bounding_box) {
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 3;
        const box = grid.bounding_box;
        
        ctx.strokeRect(
          box.left * scaleX,
          box.top * scaleY,
          (box.right - box.left) * scaleX,
          (box.bottom - box.top) * scaleY
        );
      }

      // Draw vertical center line (MAGENTA)
      if (grid.vertical_center) {
        ctx.strokeStyle = "#FF00FF";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(
          grid.vertical_center.x * scaleX,
          grid.vertical_center.y1 * scaleY
        );
        ctx.lineTo(
          grid.vertical_center.x * scaleX,
          grid.vertical_center.y2 * scaleY
        );
        ctx.stroke();
      }

      // Draw horizontal lines (thirds) with labels
      if (grid.horizontal_lines) {
        grid.horizontal_lines.forEach((line, index) => {
          const colors = ["#FFD700", "#FFA500", "#FF6347"]; // Gold, Orange, Tomato
          ctx.strokeStyle = colors[index % colors.length];
          ctx.lineWidth = 2;

          ctx.beginPath();
          ctx.moveTo(line.x1 * scaleX, line.y * scaleY);
          ctx.lineTo(line.x2 * scaleX, line.y * scaleY);
          ctx.stroke();

          // Label the line
          if (line.label) {
            ctx.fillStyle = colors[index % colors.length];
            ctx.font = "bold 12px monospace";
            ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
            ctx.shadowBlur = 3;
            ctx.fillText(
              line.label,
              (line.x2 * scaleX) + 8,
              line.y * scaleY - 4
            );
            ctx.shadowBlur = 0;
          }
        });
      }

      // Draw eye line (CYAN)
      if (grid.eye_line) {
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(grid.eye_line.x1 * scaleX, grid.eye_line.y * scaleY);
        ctx.lineTo(grid.eye_line.x2 * scaleX, grid.eye_line.y * scaleY);
        ctx.stroke();

        // Add label
        ctx.fillStyle = "#00FFFF";
        ctx.font = "bold 12px monospace";
        ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
        ctx.shadowBlur = 3;
        ctx.fillText(
          "Eye Line",
          (grid.eye_line.x2 * scaleX) + 8,
          grid.eye_line.y * scaleY - 4
        );
        ctx.shadowBlur = 0;
      }
    }
  }, [showGrid]);

  const connectWebSocket = useCallback(() => {
    try {
      console.log("Connecting WebSocket...");
      const ws = createRealtimeWebSocket();
      wsRef.current = ws;

      // Keep-alive mechanism - prevent premature disconnection
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          // Just check connection state, don't send anything extra
          console.log("WebSocket alive");
        }
      }, 30000); // Every 30 seconds

      pingIntervalRef.current = pingInterval;

      ws.onopen = () => {
        console.log("WebSocket connected");
        setIsConnected(true);
        setPermissionError(false);
      };

      ws.onmessage = (event) => {
        try {
          const data: RealtimeGridData = JSON.parse(event.data);
          
          if (data.status === "success") {
            setGridData(data);
            drawGrid(data);
          } else if (data.status === "no_face") {
            // Clear grid when no face detected
            if (canvasRef.current) {
              const ctx = canvasRef.current.getContext("2d");
              ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
            setGridData({ status: "no_face" });
          }

          // FPS counter
          fpsCounterRef.current.frames++;
          const now = Date.now();
          if (now - fpsCounterRef.current.lastTime >= 1000) {
            setFps(fpsCounterRef.current.frames);
            fpsCounterRef.current.frames = 0;
            fpsCounterRef.current.lastTime = now;
          }
        } catch (error) {
          console.error("WebSocket message error:", error);
        }
      };

      ws.onclose = (event) => {
        console.log("WebSocket disconnected", event.code, event.reason);
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        setIsConnected(false);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        setIsConnected(false);
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      setIsConnected(false);
    }
  }, [drawGrid]);

  const sendFrame = useCallback(() => {
    if (
      !webcamRef.current ||
      !wsRef.current ||
      wsRef.current.readyState !== WebSocket.OPEN
    ) {
      return;
    }

    const canvas = webcamRef.current.getCanvas();
    if (canvas) {
      canvas.toBlob(
        (blob) => {
          if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
            frameCountRef.current++;
            wsRef.current.send(blob);
          }
        },
        "image/jpeg",
        0.8
      );
    }

    animFrameRef.current = requestAnimationFrame(sendFrame);
  }, []);

  const startCamera = useCallback(() => {
    console.log("Starting camera...");
    setIsCameraOn(true);
    setPermissionError(false);
    frameCountRef.current = 0;
  }, []);

  const stopCamera = useCallback(() => {
    console.log("Stopping camera...");
    setIsCameraOn(false);
    setGridData(null);
    setFps(0);
    
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, []);

  const handleUserMedia = useCallback(() => {
    console.log("Camera ready");
    connectWebSocket();
    setTimeout(() => {
      console.log("Starting frame transmission...");
      sendFrame();
    }, 1000);
  }, [connectWebSocket, sendFrame]);

  const handleUserMediaError = useCallback((error: any) => {
    console.error("Camera error:", error);
    setPermissionError(true);
    setIsCameraOn(false);
  }, []);

  // Redraw grid when showGrid toggle changes or window resizes
  useEffect(() => {
    const handleResize = () => {
      if (gridData && gridData.status === "success") {
        drawGrid(gridData);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [gridData, drawGrid]);

  useEffect(() => {
    if (gridData && gridData.status === "success" && showGrid) {
      drawGrid(gridData);
    } else if (!showGrid && canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [showGrid, gridData, drawGrid]);

  useEffect(() => {
    return () => {
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-2xl mb-10"
          >
            <h1 className="text-4xl sm:text-5xl font-black mb-4">
              Draw <span className="gradient-text">Mode</span>
            </h1>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Use your webcam as a live reference with real-time Loomis grid
              overlay. The grid adapts to head rotation for accurate perspective
              construction.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="lg:col-span-2"
            >
              <div className="glass-card overflow-hidden">
                <div className="relative aspect-video bg-muted/30">
                  {isCameraOn ? (
                    <>
                      <Webcam
                        ref={webcamRef}
                        audio={false}
                        mirrored
                        videoConstraints={{
                          width: 1280,
                          height: 720,
                          facingMode: "user",
                        }}
                        onUserMedia={handleUserMedia}
                        onUserMediaError={handleUserMediaError}
                        className="w-full h-full object-cover"
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full pointer-events-none object-cover"
                        style={{ display: showGrid ? "block" : "none" }}
                      />
                    </>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                      <div className="w-20 h-20 rounded-3xl bg-primary/10 flex items-center justify-center">
                        <Camera className="h-10 w-10 text-primary/60" />
                      </div>
                      <div className="text-center">
                        <p className="text-lg font-semibold mb-1">Camera Off</p>
                        <p className="text-sm text-muted-foreground">
                          Click Start Camera to begin live tracking
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="p-4 flex items-center justify-between border-t border-border/30">
                  <div className="flex gap-3">
                    <button
                      onClick={isCameraOn ? stopCamera : startCamera}
                      className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
                        isCameraOn
                          ? "bg-destructive/10 text-destructive hover:bg-destructive/20"
                          : "bg-primary text-primary-foreground hover:opacity-90"
                      }`}
                    >
                      {isCameraOn ? (
                        <>
                          <CameraOff className="h-4 w-4" />
                          Stop Camera
                        </>
                      ) : (
                        <>
                          <Camera className="h-4 w-4" />
                          Start Camera
                        </>
                      )}
                    </button>
                  </div>

                  {isCameraOn && (
                    <button
                      onClick={() => setShowGrid(!showGrid)}
                      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
                        showGrid
                          ? "bg-primary/10 text-primary"
                          : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {showGrid ? (
                        <>
                          <Eye className="h-4 w-4" />
                          Grid On
                        </>
                      ) : (
                        <>
                          <EyeOff className="h-4 w-4" />
                          Grid Off
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>

              {permissionError && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 glass-card p-5 border-destructive/50 flex items-start gap-3"
                >
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-sm">Camera Access Denied</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Please allow camera access in your browser settings to use
                      Draw Mode.
                    </p>
                  </div>
                </motion.div>
              )}
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h2 className="text-xl font-bold mb-4">Live Data</h2>
              <LiveInfoPanel
                data={gridData}
                isConnected={isConnected}
                fps={fps}
              />
            </motion.div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default DrawMode;
