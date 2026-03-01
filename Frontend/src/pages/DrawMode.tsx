import { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import Webcam from "react-webcam";
import { Camera, CameraOff, Eye, EyeOff, AlertCircle } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import LiveInfoPanel from "@/components/LiveInfoPanel";
import { createRealtimeWebSocket, RealtimeGridData } from "@/lib/api";

const DrawMode = () => {
  const webcamRef          = useRef<Webcam>(null);
  const wsRef              = useRef<WebSocket | null>(null);
  const animFrameRef       = useRef<number>(0);
  const pingIntervalRef    = useRef<NodeJS.Timeout | null>(null);
  const fpsCounterRef      = useRef({ frames: 0, lastTime: Date.now() });
  const frameCountRef      = useRef(0);
  const showGridRef        = useRef(true);
  const isRunningRef       = useRef(false);
  const waitingResponseRef = useRef(false);
  const lastSentTimeRef    = useRef(0);

  const [isCameraOn, setIsCameraOn]           = useState(false);
  const [showGrid, setShowGrid]               = useState(true);
  const [isConnected, setIsConnected]         = useState(false);
  const [gridData, setGridData]               = useState<RealtimeGridData | null>(null);
  const [fps, setFps]                         = useState(0);
  const [permissionError, setPermissionError] = useState(false);
  const [annotatedFrame, setAnnotatedFrame]   = useState<string | null>(null);

  const [poseData, setPoseData] = useState<{
    pitch: number; yaw: number; roll: number; view_type: string;
  } | null>(null);

  const [measurements, setMeasurements] = useState<{
    face_width: number; face_height: number;
    eye_distance: number; nose_to_chin: number;
    mouth_width: number; nose_width: number;
  } | null>(null);

  const [ratios, setRatios] = useState<{
    eye_to_face_width: number; nose_to_face_height: number;
    face_aspect_ratio: number; mouth_to_face_width: number;
  } | null>(null);

  const [analysis, setAnalysis] = useState<{
    overall_score: number;
    face_shape: string;
    comparisons: Record<string, { detected: number; ideal: number; score: number }>;
  } | null>(null);

  // ── Keep showGridRef in sync + notify backend ───────────────────────
  useEffect(() => {
    showGridRef.current = showGrid;
    if (!showGrid) setAnnotatedFrame(null);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ grid: showGrid }));
    }
  }, [showGrid]);

  // ── WebSocket ───────────────────────────────────────────────────────
  const connectWebSocket = useCallback(() => {
    try {
      const ws = createRealtimeWebSocket();
      wsRef.current = ws;

      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) console.log("WebSocket alive");
      }, 30000);
      pingIntervalRef.current = pingInterval;

      ws.onopen = () => {
        setIsConnected(true);
        setPermissionError(false);
        ws.send(JSON.stringify({ grid: showGridRef.current }));
      };

      ws.onmessage = (event) => {
        try {
          waitingResponseRef.current = false;

          const data: RealtimeGridData & {
            frame?:        string;
            pose?:         { pitch: number; yaw: number; roll: number };
            view_type?:    string;
            measurements?: {
              face_width: number; face_height: number;
              eye_distance: number; nose_to_chin: number;
              mouth_width: number; nose_width: number;
            };
            ratios?: {
              eye_to_face_width: number; nose_to_face_height: number;
              face_aspect_ratio: number; mouth_to_face_width: number;
            };
            analysis?: {
              overall_score: number;
              face_shape: string;
              comparisons: Record<string, { detected: number; ideal: number; score: number }>;
            };
          } = JSON.parse(event.data);

          if (data.status === "ping") return;

          // Always show frame (annotated or raw)
          if (data.frame) {
            setAnnotatedFrame(`data:image/jpeg;base64,${data.frame}`);
          }

          if (data.status === "success" || data.status === "no_face") {
            setGridData(data);
          }

          if (data.pose && data.view_type) {
            setPoseData({
              pitch:     data.pose.pitch,
              yaw:       data.pose.yaw,
              roll:      data.pose.roll,
              view_type: data.view_type,
            });
          }

          if (data.measurements) setMeasurements(data.measurements);
          if (data.ratios)       setRatios(data.ratios);
          if (data.analysis)     setAnalysis(data.analysis);

          // FPS counter
          fpsCounterRef.current.frames++;
          const now = Date.now();
          if (now - fpsCounterRef.current.lastTime >= 1000) {
            setFps(fpsCounterRef.current.frames);
            fpsCounterRef.current.frames  = 0;
            fpsCounterRef.current.lastTime = now;
          }
        } catch (err) {
          waitingResponseRef.current = false;
          console.error("WebSocket message error:", err);
        }
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed", event.code, event.reason);
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        waitingResponseRef.current = false;
        setIsConnected(false);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }
        waitingResponseRef.current = false;
        setIsConnected(false);
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      setIsConnected(false);
    }
  }, []);

  // ── Frame sender ────────────────────────────────────────────────────
  const sendFrame = useCallback(() => {
    if (!isRunningRef.current) return;

    const now          = Date.now();
    const MIN_INTERVAL = showGridRef.current ? 100 : 60;

    if (
      !waitingResponseRef.current &&
      now - lastSentTimeRef.current >= MIN_INTERVAL &&
      webcamRef.current &&
      wsRef.current &&
      wsRef.current.readyState === WebSocket.OPEN
    ) {
      const canvas = webcamRef.current.getCanvas();
      if (canvas) {
        canvas.toBlob(
          (blob) => {
            if (
              blob &&
              wsRef.current?.readyState === WebSocket.OPEN &&
              !waitingResponseRef.current
            ) {
              frameCountRef.current++;
              waitingResponseRef.current = true;
              lastSentTimeRef.current    = Date.now();
              wsRef.current.send(blob);
            }
          },
          "image/jpeg",
          0.7
        );
      }
    }

    animFrameRef.current = requestAnimationFrame(sendFrame);
  }, []);

  // ── Camera controls ─────────────────────────────────────────────────
  const startCamera = useCallback(() => {
    isRunningRef.current       = true;
    waitingResponseRef.current = false;
    lastSentTimeRef.current    = 0;
    setIsCameraOn(true);
    setPermissionError(false);
    setAnnotatedFrame(null);
    setPoseData(null);
    setMeasurements(null);
    setRatios(null);
    setAnalysis(null);
    frameCountRef.current = 0;
  }, []);

  const stopCamera = useCallback(() => {
    isRunningRef.current       = false;
    waitingResponseRef.current = false;
    setIsCameraOn(false);
    setGridData(null);
    setAnnotatedFrame(null);
    setPoseData(null);
    setMeasurements(null);
    setRatios(null);
    setAnalysis(null);
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
  }, []);

  const handleUserMedia = useCallback(() => {
    connectWebSocket();
    setTimeout(() => sendFrame(), 1000);
  }, [connectWebSocket, sendFrame]);

  const handleUserMediaError = useCallback((error: unknown) => {
    console.error("Camera error:", error);
    setPermissionError(true);
    setIsCameraOn(false);
  }, []);

  // ── Cleanup on unmount ───────────────────────────────────────────────
  useEffect(() => {
    return () => {
      isRunningRef.current       = false;
      waitingResponseRef.current = false;
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
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
            {/* ── Left: Video + All Panels ── */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="lg:col-span-2 flex flex-col gap-4"
            >
              {/* Video card */}
              <div className="glass-card overflow-hidden">
                <div className="relative aspect-video bg-muted/30">
                  {isCameraOn ? (
                    <>
                      <Webcam
                        ref={webcamRef}
                        audio={false}
                        mirrored
                        videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
                        onUserMedia={handleUserMedia}
                        onUserMediaError={handleUserMediaError}
                        className={annotatedFrame ? "hidden" : "w-full h-full object-cover"}
                      />
                      {annotatedFrame ? (
                        <img
                          src={annotatedFrame}
                          alt="Live feed with Loomis grid"
                          className="w-full h-full object-cover"
                          style={{ imageRendering: "crisp-edges" }}
                        />
                      ) : (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <p className="text-sm text-muted-foreground animate-pulse">
                            Detecting face…
                          </p>
                        </div>
                      )}
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
                  <button
                    onClick={isCameraOn ? stopCamera : startCamera}
                    className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
                      isCameraOn
                        ? "bg-destructive/10 text-destructive hover:bg-destructive/20"
                        : "bg-primary text-primary-foreground hover:opacity-90"
                    }`}
                  >
                    {isCameraOn
                      ? <><CameraOff className="h-4 w-4" />Stop Camera</>
                      : <><Camera className="h-4 w-4" />Start Camera</>}
                  </button>

                  {isCameraOn && (
                    <button
                      onClick={() => setShowGrid((prev) => !prev)}
                      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
                        showGrid
                          ? "bg-primary/10 text-primary"
                          : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {showGrid
                        ? <><Eye className="h-4 w-4" />Grid On</>
                        : <><EyeOff className="h-4 w-4" />Grid Off</>}
                    </button>
                  )}
                </div>
              </div>

              {/* ── Head Pose Table ── */}
              {poseData && isCameraOn && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5"
                >
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                    Head Pose &amp; View
                  </h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border/40">
                        <th className="text-left py-2 px-3 font-medium text-muted-foreground">Metric</th>
                        <th className="text-right py-2 px-3 font-medium text-muted-foreground">Value</th>
                        <th className="text-right py-2 px-3 font-medium text-muted-foreground">Meaning</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/20">
                      <tr>
                        <td className="py-2 px-3 font-medium">Pitch</td>
                        <td className="py-2 px-3 text-right font-mono text-primary">{poseData.pitch.toFixed(1)}°</td>
                        <td className="py-2 px-3 text-right text-muted-foreground text-xs">
                          {poseData.pitch > 5 ? "Looking Down" : poseData.pitch < -5 ? "Looking Up" : "Level"}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-medium">Yaw</td>
                        <td className="py-2 px-3 text-right font-mono text-primary">{poseData.yaw.toFixed(1)}°</td>
                        <td className="py-2 px-3 text-right text-muted-foreground text-xs">
                          {poseData.yaw > 5 ? "Turned Right" : poseData.yaw < -5 ? "Turned Left" : "Center"}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-medium">Roll</td>
                        <td className="py-2 px-3 text-right font-mono text-primary">{poseData.roll.toFixed(1)}°</td>
                        <td className="py-2 px-3 text-right text-muted-foreground text-xs">
                          {poseData.roll > 5 ? "Tilted Right" : poseData.roll < -5 ? "Tilted Left" : "Straight"}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-medium">View</td>
                        <td colSpan={2} className="py-2 px-3 text-right font-medium text-purple-400">
                          {poseData.view_type}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </motion.div>
              )}

              {/* ── Detailed Measurements ── */}
              {measurements && isCameraOn && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5"
                >
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                    Detailed Measurements
                  </h3>
                  <div className="space-y-1">
                    {[
                      {
                        label: "Face Dimensions",
                        value: `${measurements.face_width.toFixed(0)} × ${measurements.face_height.toFixed(0)} px`,
                        sub:   `${(measurements.face_width * 0.0264).toFixed(1)} × ${(measurements.face_height * 0.0264).toFixed(1)} cm`,
                      },
                      {
                        label: "Eye Distance",
                        value: `${measurements.eye_distance.toFixed(0)} px`,
                        sub:   `${(measurements.eye_distance * 0.0264).toFixed(1)} cm`,
                      },
                      {
                        label: "Nose to Chin",
                        value: `${measurements.nose_to_chin.toFixed(0)} px`,
                        sub:   `${(measurements.nose_to_chin * 0.0264).toFixed(1)} cm`,
                      },
                      {
                        label: "Mouth Width",
                        value: `${measurements.mouth_width.toFixed(0)} px`,
                        sub:   `${(measurements.mouth_width * 0.0264).toFixed(1)} cm`,
                      },
                      {
                        label: "Nose Width",
                        value: `${measurements.nose_width.toFixed(0)} px`,
                        sub:   `${(measurements.nose_width * 0.0264).toFixed(1)} cm`,
                      },
                    ].map((row) => (
                      <div
                        key={row.label}
                        className="flex items-center justify-between py-2 border-b border-border/20 last:border-0"
                      >
                        <span className="text-sm text-muted-foreground">{row.label}</span>
                        <div className="text-right">
                          <span className="text-sm font-mono font-medium">{row.value}</span>
                          <span className="text-xs text-muted-foreground ml-2">({row.sub})</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* ── Proportional Ratios ── */}
              {ratios && isCameraOn && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5"
                >
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
                    Proportional Ratios
                  </h3>
                  <div className="space-y-4">
                    {[
                      { label: "Eye to Face Width",   value: ratios.eye_to_face_width,   ideal: 0.46 },
                      { label: "Nose to Face Height", value: ratios.nose_to_face_height, ideal: 0.33 },
                      { label: "Face Aspect Ratio",   value: ratios.face_aspect_ratio,   ideal: 0.75 },
                      { label: "Mouth to Face Width", value: ratios.mouth_to_face_width, ideal: 0.46 },
                    ].map((row) => {
                      const diff     = Math.abs(row.value - row.ideal);
                      const pct      = Math.max(0, Math.min(100, 100 - diff * 200));
                      const barColor = pct >= 85 ? "bg-green-500" : pct >= 60 ? "bg-yellow-500" : "bg-red-500";
                      return (
                        <div key={row.label}>
                          <div className="flex justify-between text-sm mb-1.5">
                            <span className="text-muted-foreground">{row.label}</span>
                            <span className="font-mono font-medium">{row.value.toFixed(3)}</span>
                          </div>
                          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-xs text-muted-foreground mt-4">
                    Ratios compared against classical Loomis ideal proportions
                  </p>
                </motion.div>
              )}

              {/* ── Proportion Analysis ── */}
              {analysis && isCameraOn && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                      Proportion Analysis
                    </h3>
                    <span className="text-xs font-semibold text-primary bg-primary/10 px-2.5 py-1 rounded-full">
                      {analysis.face_shape}
                    </span>
                  </div>
                  <div className="space-y-4">
                    {Object.entries(analysis.comparisons).map(([key, val]) => {
                      const label =
                        key === "eye_spacing"     ? "Eye Spacing"      :
                        key === "nose_chin_ratio" ? "Nose-Chin Ratio"  :
                        key === "face_aspect"     ? "Face Aspect Ratio": key;
                      const scoreColor =
                        val.score >= 85 ? "text-green-400" :
                        val.score >= 60 ? "text-yellow-400" : "text-red-400";
                      return (
                        <div key={key} className="flex items-start justify-between">
                          <div>
                            <p className="text-sm font-medium">{label}</p>
                            <p className="text-xs text-muted-foreground mt-0.5">
                              Detected: {val.detected.toFixed(3)} | Ideal: {val.ideal.toFixed(3)}
                            </p>
                          </div>
                          <span className={`text-lg font-bold ${scoreColor}`}>
                            {val.score.toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-4 pt-3 border-t border-border/30 flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Overall Score</span>
                    <span className={`text-2xl font-black ${
                      analysis.overall_score >= 85 ? "text-green-400" :
                      analysis.overall_score >= 60 ? "text-yellow-400" : "text-red-400"
                    }`}>
                      {analysis.overall_score.toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              )}

              {/* ── Permission Error ── */}
              {permissionError && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5 border-destructive/50 flex items-start gap-3"
                >
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-sm">Camera Access Denied</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Please allow camera access in your browser settings to use Draw Mode.
                    </p>
                  </div>
                </motion.div>
              )}
            </motion.div>

            {/* ── Right: Live Info Panel ── */}
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
