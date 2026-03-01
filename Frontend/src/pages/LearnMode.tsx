import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Loader2, AlertCircle, Image as ImageIcon, Maximize2, Minimize2 } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ImageUploader from "@/components/ImageUploader";
import TutorialSteps from "@/components/TutorialSteps";
import AnalysisDashboard from "@/components/AnalysisDashboard";
import { processTutorial, processImage, generateSketchCanvas } from "@/lib/api";
import type { TutorialResult, ProcessResult, SketchCanvasResult } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

// ── Canvas size presets (real-world cm) ──────────────────────
const CANVAS_SIZES = [
  { label: "A4",       w: 21.0, h: 29.7 },
  { label: "A3",       w: 29.7, h: 42.0 },
  { label: "20×25 cm", w: 20.0, h: 25.0 },
  { label: "30×40 cm", w: 30.0, h: 40.0 },
  { label: "40×50 cm", w: 40.0, h: 50.0 },
  { label: "60×80 cm", w: 60.0, h: 80.0 },
];

// ── px → cm converter ────────────────────────────────────────
const computeCmMeasurements = (
  measurements_px: Record<string, number>,
  canvas: { w: number; h: number }
) => {
  const faceHeightPx =
    Number(measurements_px["face_height"]) ||
    Number(measurements_px["Face Height"]) ||
    226;
  if (faceHeightPx <= 0) return [];
  const pxPerCm = faceHeightPx / (canvas.h * 0.7);
  return Object.entries(measurements_px).map(([key, rawVal]) => {
    const px = Number(rawVal);
    return {
      label: key.replace(/_/g, " "),
      px:    px.toFixed(1),
      cm:    pxPerCm > 0 ? (px / pxPerCm).toFixed(2) : "—",
    };
  });
};

const LearnMode = () => {
  const [file, setFile]                     = useState<File | null>(null);
  const [isProcessing, setIsProcessing]     = useState(false);
  const [tutorialResult, setTutorialResult] = useState<TutorialResult | null>(null);
  const [processResult, setProcessResult]   = useState<ProcessResult | null>(null);
  const [activeStep, setActiveStep]         = useState(0);
  const [error, setError]                   = useState<string | null>(null);
  const { toast }                           = useToast();

  // Sketch Canvas state
  const [sketchCanvasOpen, setSketchCanvasOpen]     = useState(false);
  const [sketchCanvasResult, setSketchCanvasResult] = useState<SketchCanvasResult | null>(null);
  const [sketchLoading, setSketchLoading]           = useState(false);
  const [canvasSize, setCanvasSize]                 = useState(CANVAS_SIZES[0]);

  // Fullscreen state for grid column
  const [gridFullscreen, setGridFullscreen] = useState(false);
  const gridColumnRef                       = useRef<HTMLDivElement>(null);

  const totalSteps = tutorialResult?.tutorial_steps?.length ?? 0;

  // ── Keyboard arrow navigation ────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only navigate when tutorial is loaded and sketch modal is NOT open
      if (!tutorialResult || sketchCanvasOpen) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setActiveStep((prev) => Math.min(prev + 1, totalSteps - 1));
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setActiveStep((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Escape" && gridFullscreen) {
        setGridFullscreen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [tutorialResult, totalSteps, sketchCanvasOpen, gridFullscreen]);

  // ── Handlers ──────────────────────────────────────────────────
  const handleFileSelect = useCallback((selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setTutorialResult(null);
    setProcessResult(null);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError(null);
    setActiveStep(0);
    try {
      const [tutorial, analysis] = await Promise.all([
        processTutorial(file),
        processImage(file),
      ]);
      setTutorialResult(tutorial);
      setProcessResult(analysis);
      const faceShape = analysis.faces?.[0]?.analysis?.face_shape || "Unknown";
      const stepCount = tutorial.tutorial_steps?.length || 0;
      toast({
        title: "Analysis Complete",
        description: `Detected ${faceShape} face with ${stepCount} tutorial steps.`,
      });
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || "Failed to process image";
      setError(message);
      toast({
        title: "Processing Error",
        description: "Make sure the backend server is running on localhost:8000",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setTutorialResult(null);
    setProcessResult(null);
    setActiveStep(0);
    setError(null);
    setGridFullscreen(false);
  };

  const handleSketchCanvas = async () => {
    if (!file) return;
    setSketchLoading(true);
    setSketchCanvasOpen(true);
    setSketchCanvasResult(null);
    try {
      const result = await generateSketchCanvas(file);
      setSketchCanvasResult(result);
    } catch (err: any) {
      toast({
        title: "Sketch Canvas Error",
        description: err.response?.data?.detail || "Failed to generate sketch canvas",
        variant: "destructive",
      });
      setSketchCanvasOpen(false);
    } finally {
      setSketchLoading(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">

          {/* Page Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-2xl mb-10"
          >
            <h1 className="text-4xl sm:text-5xl font-black mb-4">
              Learn <span className="gradient-text">Mode</span>
            </h1>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Upload a portrait and receive a step-by-step Loomis grid construction
              tutorial with detailed proportion analysis.
            </p>
          </motion.div>

          {/* ════════════════════════════════════════════════════
              3-COLUMN LAYOUT
              Col 1 (3/12) — Upload + Buttons
              Col 2 (5/12) — Grid Steps (fullscreen-capable)
              Col 3 (4/12) — Analysis Dashboard
          ════════════════════════════════════════════════════ */}
          <div className="grid lg:grid-cols-12 gap-6 items-start">

            {/* ── COLUMN 1: Upload & Controls ── */}
            <div className={`lg:col-span-3 space-y-5 ${gridFullscreen ? "hidden" : ""}`}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <ImageUploader
                  onFileSelect={handleFileSelect}
                  isProcessing={isProcessing}
                />
              </motion.div>

              {file && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                  className="flex flex-col gap-3"
                >
                  {!tutorialResult ? (
                    <button
                      onClick={handleAnalyze}
                      disabled={isProcessing}
                      className="w-full py-4 rounded-2xl bg-primary text-primary-foreground
                                 font-semibold text-base transition-all duration-300
                                 hover:opacity-90 disabled:opacity-50
                                 flex items-center justify-center gap-3 btn-glow"
                    >
                      {isProcessing ? (
                        <><Loader2 className="h-5 w-5 animate-spin" />Analyzing...</>
                      ) : (
                        <><ImageIcon className="h-5 w-5" />Analyze Portrait</>
                      )}
                    </button>
                  ) : (
                    <button
                      onClick={handleReset}
                      className="w-full py-4 rounded-2xl bg-secondary text-secondary-foreground
                                 font-semibold text-base transition-all
                                 hover:bg-secondary/80"
                    >
                      Upload New Image
                    </button>
                  )}

                  <button
                    onClick={handleSketchCanvas}
                    disabled={sketchLoading}
                    className="w-full py-3 rounded-2xl border-2 border-primary/40 text-primary
                               font-semibold text-base transition-all duration-300
                               hover:bg-primary/10 disabled:opacity-50
                               flex items-center justify-center gap-2"
                  >
                    {sketchLoading
                      ? <Loader2 className="h-5 w-5 animate-spin" />
                      : <>🖊️ Sketch Canvas</>
                    }
                  </button>
                </motion.div>
              )}

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-4 border-destructive/50 flex items-start gap-3"
                >
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-sm">Error</p>
                    <p className="text-sm text-muted-foreground mt-1">{error}</p>
                  </div>
                </motion.div>
              )}
            </div>

            {/* ── COLUMN 2: Grid Step Images — fullscreen-capable ── */}
            <div
              ref={gridColumnRef}
              className={
                gridFullscreen
                  ? "fixed inset-0 z-40 bg-background flex flex-col p-6 overflow-auto"
                  : "lg:col-span-5"
              }
            >
              {tutorialResult && tutorialResult.tutorial_steps ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="h-full flex flex-col"
                >
                  {/* ── Header bar: keyboard hint + fullscreen button ── */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground bg-secondary/50
                                       px-2.5 py-1 rounded-lg">
                        ← → Arrow keys to navigate
                      </span>
                      <span className="text-xs font-semibold text-muted-foreground
                                       bg-primary/10 text-primary px-2.5 py-1 rounded-lg">
                        {activeStep + 1} / {totalSteps}
                      </span>
                    </div>

                    {/* Fullscreen toggle */}
                    <button
                      onClick={() => setGridFullscreen((v) => !v)}
                      className="p-2 rounded-xl border border-border/50
                                 text-muted-foreground hover:text-primary
                                 hover:border-primary/40 transition-all"
                      title={gridFullscreen ? "Exit fullscreen (Esc)" : "Fullscreen"}
                    >
                      {gridFullscreen
                        ? <Minimize2 className="h-4 w-4" />
                        : <Maximize2 className="h-4 w-4" />
                      }
                    </button>
                  </div>

                  {/* Grid image — NO duplicate tab row here, TutorialSteps handles its own */}
                  <div className={`glass-card overflow-hidden rounded-2xl flex-1
                                   ${gridFullscreen ? "flex flex-col" : ""}`}>
                    <TutorialSteps
                      steps={tutorialResult.tutorial_steps}
                      activeStep={activeStep}
                      onStepChange={setActiveStep}
                    />
                  </div>

                  {/* Fullscreen Esc hint */}
                  {gridFullscreen && (
                    <p className="text-center text-xs text-muted-foreground mt-3">
                      Press <kbd className="px-1.5 py-0.5 rounded bg-secondary text-xs">Esc</kbd> or
                      click <Minimize2 className="inline h-3 w-3 mx-1" /> to exit fullscreen
                    </p>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="glass-card p-10 text-center flex flex-col
                             items-center justify-center min-h-[320px]"
                >
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-primary/10
                                  flex items-center justify-center mb-4">
                    <ImageIcon className="h-8 w-8 text-primary/60" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Grid Steps</h3>
                  <p className="text-sm text-muted-foreground">
                    Upload an image and click Analyze to see the 6-step
                    Loomis grid construction.
                  </p>
                </motion.div>
              )}
            </div>

            {/* ── COLUMN 3: Analysis Dashboard — completely independent ── */}
            <div className={`lg:col-span-4 ${gridFullscreen ? "hidden" : ""}`}>
              {processResult && processResult.faces && processResult.faces.length > 0 ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="space-y-5"
                >
                  <h2 className="text-xl font-bold">Analysis Results</h2>
                  <AnalysisDashboard result={processResult} />

                  {processResult.processed_image && (
                    <div className="glass-card overflow-hidden rounded-2xl">
                      <img
                        src={`data:image/jpeg;base64,${processResult.processed_image}`}
                        alt="Processed with grid overlay"
                        className="w-full object-contain"
                      />
                      <div className="p-4 border-t border-border/30">
                        <p className="text-sm font-medium">Complete Grid Overlay</p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {processResult.face_count} face
                          {processResult.face_count !== 1 ? "s" : ""} detected
                          with Loomis grid construction
                        </p>
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                  className="glass-card p-10 text-center flex flex-col
                             items-center justify-center min-h-[320px]"
                >
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-primary/10
                                  flex items-center justify-center mb-4">
                    <ImageIcon className="h-8 w-8 text-primary/60" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No Analysis Yet</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Upload a portrait and click Analyze to see proportion
                    measurements and face shape classification.
                  </p>
                </motion.div>
              )}
            </div>

          </div>
          {/* ══ END 3-COLUMN LAYOUT ══ */}

        </div>
      </main>

      <Footer />

      {/* ══════════════════════════════════════════════
          SKETCH CANVAS MODAL
      ══════════════════════════════════════════════ */}
      {sketchCanvasOpen && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
          <div className="bg-background rounded-2xl shadow-2xl w-full max-w-5xl
                          max-h-[95vh] flex flex-col border border-border">

            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4
                            border-b border-border shrink-0">
              <h2 className="text-xl font-bold">🖊️ Loomis Sketch Canvas</h2>
              <button
                onClick={() => setSketchCanvasOpen(false)}
                className="text-muted-foreground hover:text-destructive
                           text-2xl font-bold px-2 transition-colors"
              >✕</button>
            </div>

            {/* Canvas Size Selector */}
            <div className="px-6 py-3 border-b border-border shrink-0
                            flex items-center gap-3 flex-wrap">
              <span className="text-sm font-semibold text-muted-foreground shrink-0">
                🎨 Canvas Size:
              </span>
              {CANVAS_SIZES.map((size) => (
                <button
                  key={size.label}
                  onClick={() => setCanvasSize(size)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all
                    ${canvasSize.label === size.label
                      ? "bg-primary text-primary-foreground border-primary shadow-sm"
                      : "border-border text-muted-foreground hover:border-primary hover:text-primary"
                    }`}
                >
                  {size.label}
                </button>
              ))}
              <span className="text-xs text-muted-foreground ml-1">
                → {canvasSize.w} × {canvasSize.h} cm
              </span>
            </div>

            {/* Modal Body — strict two-column */}
            <div className="flex flex-1 min-h-0">

              {/* LEFT: Canvas image */}
              <div className="flex-1 flex items-center justify-center
                              bg-white dark:bg-muted/10 border-r border-border
                              p-6 overflow-auto">
                {sketchLoading ? (
                  <div className="flex flex-col items-center gap-3 text-muted-foreground">
                    <Loader2 className="h-10 w-10 animate-spin text-primary" />
                    <p className="text-sm font-medium">Generating sketch canvas...</p>
                    <p className="text-xs text-muted-foreground">
                      Mapping 468 landmarks to canvas...
                    </p>
                  </div>
                ) : sketchCanvasResult ? (
                  <img
                    src={sketchCanvasResult.canvas_image}
                    alt="Loomis Sketch Canvas"
                    className="max-w-full object-contain rounded-lg shadow-sm"
                    style={{ maxHeight: "calc(95vh - 160px)" }}
                  />
                ) : null}
              </div>

              {/* RIGHT: Measurements panel */}
              <div className="w-72 shrink-0 overflow-y-auto p-5 space-y-5">
                {sketchCanvasResult ? (
                  <>
                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-widest
                                     text-muted-foreground mb-3 border-b border-border pb-1">
                        📐 Measurements — {canvasSize.label}
                      </h3>
                      {computeCmMeasurements(
                        sketchCanvasResult.ratios.measurements_px,
                        canvasSize
                      ).map(({ label, px, cm }) => (
                        <div key={label}
                             className="flex justify-between items-center py-2
                                        border-b border-border/30 last:border-0">
                          <span className="text-sm text-muted-foreground capitalize">
                            {label}
                          </span>
                          <div className="text-right leading-tight">
                            <span className="text-sm font-bold font-mono text-foreground">
                              {cm} cm
                            </span>
                            <br />
                            <span className="text-xs text-muted-foreground">{px} px</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-widest
                                     text-muted-foreground mb-3 border-b border-border pb-1">
                        📊 Proportional Ratios
                      </h3>
                      {Object.entries(sketchCanvasResult.ratios.proportional_ratios).map(
                        ([key, val]) => (
                          <div key={key}
                               className="flex justify-between items-center py-2
                                          border-b border-border/30 last:border-0">
                            <span className="text-sm text-muted-foreground capitalize">
                              {key.replace(/_/g, " ")}
                            </span>
                            <span className="text-sm font-mono font-bold">{val}</span>
                          </div>
                        )
                      )}
                    </div>

                    <div className="text-center py-4 rounded-xl
                                    bg-primary/5 border border-primary/10">
                      <h3 className="text-xs font-bold uppercase tracking-widest
                                     text-muted-foreground mb-2">
                        🎯 Proportion Score
                      </h3>
                      <div className="text-5xl font-black text-primary leading-none">
                        {sketchCanvasResult.analysis.overall_score}
                        <span className="text-lg font-normal text-muted-foreground"> / 100</span>
                      </div>
                      <p className="text-sm font-semibold mt-2 text-foreground">
                        {sketchCanvasResult.analysis.face_shape}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <h3 className="text-xs font-bold uppercase tracking-widest
                                     text-muted-foreground border-b border-border pb-1">
                        💡 Recommendations
                      </h3>
                      {sketchCanvasResult.analysis.recommendations.map((r, i) => (
                        <p key={i}
                           className="text-xs text-muted-foreground bg-primary/5
                                      rounded-lg px-3 py-2 border border-primary/10">
                          • {r}
                        </p>
                      ))}
                    </div>

                    <a
                      href={sketchCanvasResult.canvas_image}
                      download="loomis_sketch_canvas.png"
                      className="block text-center bg-primary text-primary-foreground
                                 px-4 py-3 rounded-xl text-sm font-semibold
                                 hover:opacity-90 transition-opacity"
                    >
                      ⬇️ Download Canvas
                    </a>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full
                                  text-sm text-muted-foreground">
                    {sketchLoading ? "Generating..." : "No data yet"}
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
};

export default LearnMode;
