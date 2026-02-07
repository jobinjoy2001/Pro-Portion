import { motion } from "framer-motion";
import { Download, ChevronRight } from "lucide-react";
import { TutorialStep as TutorialStepType } from "@/lib/api";
import { getTutorialDownloadUrl } from "@/lib/api";

interface TutorialStepsProps {
  steps: TutorialStepType[];
  activeStep: number;
  onStepChange: (step: number) => void;
}

const stepDescriptions = [
  "Establishing the face bounding box — the foundation of all proportions",
  "Drawing the vertical centerline to divide the face symmetrically",
  "Marking horizontal thirds — hairline, eyebrow line, and nose base",
  "Placing the eye line at the midpoint of the face height",
  "Defining the jaw contour and chin placement",
  "Complete Loomis grid with all construction lines overlaid",
];

const TutorialSteps = ({ steps, activeStep, onStepChange }: TutorialStepsProps) => {
  return (
    <div className="space-y-6">
      {/* Step navigation */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {steps.map((step, index) => (
          <button
            key={index}
            onClick={() => onStepChange(index)}
            className={`flex-shrink-0 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 ${
              activeStep === index
                ? "bg-primary text-primary-foreground"
                : "bg-secondary/60 text-muted-foreground hover:text-foreground hover:bg-secondary"
            }`}
          >
            Step {index + 1}
          </button>
        ))}
      </div>

      {/* Active step display */}
      <motion.div
        key={activeStep}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.4 }}
        className="glass-card overflow-hidden"
      >
        <div className="relative">
          <img
            src={getTutorialDownloadUrl(steps[activeStep].filename)}
            alt={steps[activeStep].title}
            className="w-full object-contain max-h-[500px] bg-background/50"
          />
        </div>
        <div className="p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-1 rounded-md">
                  STEP {activeStep + 1} / {steps.length}
                </span>
              </div>
              <h3 className="text-xl font-bold mb-2">{steps[activeStep].title}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                {stepDescriptions[activeStep] || "Analyzing facial proportions..."}
              </p>
            </div>
            <a
              href={getTutorialDownloadUrl(steps[activeStep].filename)}
              download
              className="flex-shrink-0 p-3 rounded-xl bg-secondary/60 text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
            >
              <Download className="h-5 w-5" />
            </a>
          </div>
        </div>
      </motion.div>

      {/* Step navigation arrows */}
      <div className="flex justify-between">
        <button
          onClick={() => onStepChange(Math.max(0, activeStep - 1))}
          disabled={activeStep === 0}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-secondary/60 text-muted-foreground hover:text-foreground hover:bg-secondary transition-all disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronRight className="h-4 w-4 rotate-180" />
          Previous
        </button>
        <button
          onClick={() => onStepChange(Math.min(steps.length - 1, activeStep + 1))}
          disabled={activeStep === steps.length - 1}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-secondary/60 text-muted-foreground hover:text-foreground hover:bg-secondary transition-all disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
};

export default TutorialSteps;
