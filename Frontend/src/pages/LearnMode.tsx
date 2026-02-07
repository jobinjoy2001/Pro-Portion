import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Loader2, AlertCircle, Image as ImageIcon } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ImageUploader from "@/components/ImageUploader";
import TutorialSteps from "@/components/TutorialSteps";
import AnalysisDashboard from "@/components/AnalysisDashboard";
import { processTutorial, processImage, getDownloadUrl } from "@/lib/api";
import type { TutorialResult, ProcessResult } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const LearnMode = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [tutorialResult, setTutorialResult] = useState<TutorialResult | null>(null);
  const [processResult, setProcessResult] = useState<ProcessResult | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const handleFileSelect = useCallback((selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
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
      toast({
        title: "Analysis Complete",
        description: `Detected ${analysis.face_shape} face with ${tutorial.tutorial_steps.length} tutorial steps.`,
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

          <div className="grid lg:grid-cols-5 gap-8">
            {/* Left: Upload & Tutorial */}
            <div className="lg:col-span-3 space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <ImageUploader onFileSelect={handleFileSelect} isProcessing={isProcessing} />
              </motion.div>

              {/* Analyze Button */}
              {file && !tutorialResult && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <button
                    onClick={handleAnalyze}
                    disabled={isProcessing}
                    className="w-full py-4 rounded-2xl bg-primary text-primary-foreground font-semibold text-lg transition-all duration-300 hover:opacity-90 disabled:opacity-50 flex items-center justify-center gap-3 btn-glow"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Analyzing Portrait...
                      </>
                    ) : (
                      <>
                        <ImageIcon className="h-5 w-5" />
                        Analyze Portrait
                      </>
                    )}
                  </button>
                </motion.div>
              )}

              {/* Error */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-5 border-destructive/50 flex items-start gap-3"
                >
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-sm">Connection Error</p>
                    <p className="text-sm text-muted-foreground mt-1">{error}</p>
                  </div>
                </motion.div>
              )}

              {/* Tutorial Steps */}
              {tutorialResult && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <TutorialSteps
                    steps={tutorialResult.tutorial_steps}
                    activeStep={activeStep}
                    onStepChange={setActiveStep}
                  />
                </motion.div>
              )}
            </div>

            {/* Right: Analysis Dashboard */}
            <div className="lg:col-span-2">
              {processResult ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <h2 className="text-xl font-bold mb-4">Analysis Results</h2>
                  <AnalysisDashboard result={processResult} />

                  {/* Processed Image */}
                  {processResult.processed_image && (
                    <div className="mt-6 glass-card overflow-hidden">
                      <img
                        src={getDownloadUrl(processResult.processed_image)}
                        alt="Processed with grid overlay"
                        className="w-full object-contain"
                      />
                      <div className="p-4">
                        <p className="text-sm text-muted-foreground">Complete grid overlay</p>
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                  className="glass-card p-10 text-center"
                >
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-primary/10 flex items-center justify-center mb-5">
                    <ImageIcon className="h-8 w-8 text-primary/60" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No Analysis Yet</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Upload a portrait image and click Analyze to see proportion 
                    measurements, face shape classification, and a step-by-step 
                    Loomis grid tutorial.
                  </p>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default LearnMode;
