import { motion } from "framer-motion";
import { ProcessResult } from "@/lib/api";

interface AnalysisDashboardProps {
  result: ProcessResult;
}

const ProportionScore = ({ score }: { score: number }) => {
  const circumference = 2 * Math.PI * 45;
  const offset = circumference - (score / 100) * circumference;
  
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-32 h-32">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50" cy="50" r="45"
            fill="none"
            className="stroke-secondary"
            strokeWidth="6"
          />
          <circle
            cx="50" cy="50" r="45"
            fill="none"
            className="stroke-primary score-ring"
            strokeWidth="6"
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-3xl font-black">{score}</span>
        </div>
      </div>
      <span className="text-sm text-muted-foreground font-medium">Proportion Score</span>
    </div>
  );
};

const AnalysisDashboard = ({ result }: AnalysisDashboardProps) => {
  const analysis = result.ml_analyses?.[0];

  return (
    <div className="space-y-6">
      {/* Score and Shape */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-card p-6 flex flex-col items-center justify-center">
          <ProportionScore score={result.proportion_score} />
        </div>
        <div className="glass-card p-6 flex flex-col items-center justify-center gap-3">
          <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
            <span className="text-2xl">
              {result.face_shape === "Oval" ? "🥚" :
               result.face_shape === "Round" ? "🔵" :
               result.face_shape === "Square" ? "🟦" : "📐"}
            </span>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold">{result.face_shape}</p>
            <p className="text-sm text-muted-foreground">Face Shape</p>
          </div>
        </div>
      </div>

      {/* Measurements */}
      {analysis && (
        <div className="glass-card p-6 space-y-4">
          <h3 className="text-lg font-bold">Detailed Measurements</h3>
          <div className="grid gap-3">
            <MeasurementRow
              label="Face Dimensions"
              value={`${Math.round(analysis.face_width)} × ${Math.round(analysis.face_height)} px`}
            />
            <MeasurementRow
              label="Eye Distance"
              value={`${Math.round(analysis.eye_distance)} px`}
            />
            <MeasurementRow
              label="Nose-to-Chin Ratio"
              value={`${(analysis.nose_chin_ratio * 100).toFixed(1)}%`}
            />
          </div>
        </div>
      )}

      {/* Facial Thirds */}
      {analysis?.thirds && (
        <div className="glass-card p-6 space-y-4">
          <h3 className="text-lg font-bold">Facial Thirds</h3>
          <div className="space-y-3">
            <ThirdBar label="Upper Third" value={analysis.thirds.upper} ideal={33.3} />
            <ThirdBar label="Middle Third" value={analysis.thirds.middle} ideal={33.3} />
            <ThirdBar label="Lower Third" value={analysis.thirds.lower} ideal={33.3} />
          </div>
          <p className="text-xs text-muted-foreground">
            Classical ideal proportions divide the face into equal thirds
          </p>
        </div>
      )}
    </div>
  );
};

const MeasurementRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between py-2 border-b border-border/30 last:border-0">
    <span className="text-sm text-muted-foreground">{label}</span>
    <span className="mono-text text-foreground">{value}</span>
  </div>
);

const ThirdBar = ({ label, value, ideal }: { label: string; value: number; ideal: number }) => {
  const deviation = Math.abs(value - ideal);
  const color = deviation < 3 ? "bg-green-500" : deviation < 6 ? "bg-yellow-500" : "bg-accent";
  
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="mono-text">{value.toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-secondary overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
    </div>
  );
};

export default AnalysisDashboard;
