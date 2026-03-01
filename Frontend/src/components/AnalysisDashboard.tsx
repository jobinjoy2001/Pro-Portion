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
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-3xl font-black">{score.toFixed(1)}</span>
        </div>
      </div>
      <span className="text-sm text-muted-foreground font-medium">Proportion Score</span>
    </div>
  );
};

const AnalysisDashboard = ({ result }: AnalysisDashboardProps) => {
  // Get the first face's data
  const firstFace = result.faces?.[0];
  const measurements = firstFace?.measurements_px;
  const ratios = firstFace?.proportional_ratios;
  const analysis = firstFace?.analysis;
  
  // Extract overall score and face shape
  const overallScore = analysis?.overall_score || 0;
  const faceShape = analysis?.face_shape || "Unknown";

  return (
    <div className="space-y-6">
      {/* Score and Shape */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-card p-6 flex flex-col items-center justify-center">
          <ProportionScore score={overallScore} />
        </div>
        <div className="glass-card p-6 flex flex-col items-center justify-center gap-3">
          <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
            <span className="text-2xl">
              {faceShape.includes("Oval") || faceShape.includes("Balanced") ? "🥚" :
               faceShape.includes("Round") || faceShape.includes("Square") ? "🔵" :
               faceShape.includes("Oblong") || faceShape.includes("Long") ? "📏" : "📐"}
            </span>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold">{faceShape}</p>
            <p className="text-sm text-muted-foreground">Face Shape</p>
          </div>
        </div>
      </div>

      {/* Detailed Measurements */}
      {measurements && (
        <div className="glass-card p-6 space-y-4">
          <h3 className="text-lg font-bold">Detailed Measurements</h3>
          <div className="grid gap-3">
            <MeasurementRow
              label="Face Dimensions"
              value={`${Math.round(measurements.face_width || 0)} × ${Math.round(measurements.face_height || 0)} px`}
            />
            <MeasurementRow
              label="Eye Distance"
              value={`${Math.round(measurements.eye_distance || 0)} px`}
            />
            <MeasurementRow
              label="Nose to Chin"
              value={`${Math.round(measurements.nose_to_chin || 0)} px`}
            />
            <MeasurementRow
              label="Mouth Width"
              value={`${Math.round(measurements.mouth_width || 0)} px`}
            />
            <MeasurementRow
              label="Nose Width"
              value={`${Math.round(measurements.nose_width || 0)} px`}
            />
          </div>
        </div>
      )}

      {/* Proportional Ratios */}
      {ratios && (
        <div className="glass-card p-6 space-y-4">
          <h3 className="text-lg font-bold">Proportional Ratios</h3>
          <div className="space-y-3">
            <RatioBar
              label="Eye to Face Width"
              value={ratios.eye_to_face_width || 0}
              ideal={0.46}
              format={(v) => v.toFixed(3)}
            />
            <RatioBar
              label="Nose to Face Height"
              value={ratios.nose_to_face_height || 0}
              ideal={0.33}
              format={(v) => v.toFixed(3)}
            />
            <RatioBar
              label="Face Aspect Ratio"
              value={ratios.face_aspect_ratio || 0}
              ideal={0.75}
              format={(v) => v.toFixed(3)}
            />
            <RatioBar
              label="Mouth to Face Width"
              value={ratios.mouth_to_face_width || 0}
              ideal={0.5}
              format={(v) => v.toFixed(3)}
            />
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            Ratios are compared against classical ideal proportions (Loomis method)
          </p>
        </div>
      )}

      {/* Detailed Comparison */}
      {analysis?.comparisons && (
        <div className="glass-card p-6 space-y-4">
          <h3 className="text-lg font-bold">Proportion Analysis</h3>
          <div className="space-y-3">
            {Object.entries(analysis.comparisons).map(([key, data]: [string, any]) => (
              <ComparisonRow
                key={key}
                label={formatLabel(key)}
                detected={data.detected}
                ideal={data.ideal}
                score={data.score}
              />
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {analysis?.recommendations && analysis.recommendations.length > 0 && (
        <div className="glass-card p-6 space-y-3">
          <h3 className="text-lg font-bold">Artist Notes</h3>
          <ul className="space-y-2">
            {analysis.recommendations.map((rec: string, idx: number) => (
              <li key={idx} className="flex items-start gap-2 text-sm">
                <span className="text-primary mt-0.5">•</span>
                <span className="text-muted-foreground">{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

const MeasurementRow = ({ label, value }: { label: string; value: string }) => (
  <div className="flex items-center justify-between py-2 border-b border-border/30 last:border-0">
    <span className="text-sm text-muted-foreground">{label}</span>
    <span className="font-mono text-foreground font-medium">{value}</span>
  </div>
);

const RatioBar = ({ 
  label, 
  value, 
  ideal, 
  format 
}: { 
  label: string; 
  value: number; 
  ideal: number;
  format: (v: number) => string;
}) => {
  const deviation = Math.abs(value - ideal);
  const deviationPercent = (deviation / ideal) * 100;
  const color = deviationPercent < 10 ? "bg-green-500" : deviationPercent < 20 ? "bg-yellow-500" : "bg-red-500";
  
  // Normalize to percentage for display (0-100%)
  const displayPercent = Math.min((value / (ideal * 2)) * 100, 100);
  
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-foreground">{format(value)}</span>
      </div>
      <div className="h-2 rounded-full bg-secondary overflow-hidden relative">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${displayPercent}%` }}
          transition={{ duration: 1, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className={`h-full rounded-full ${color}`}
        />
        {/* Ideal marker */}
        <div 
          className="absolute top-0 w-0.5 h-full bg-white/50"
          style={{ left: `${Math.min((ideal / (ideal * 2)) * 100, 100)}%` }}
        />
      </div>
    </div>
  );
};

const ComparisonRow = ({ 
  label, 
  detected, 
  ideal, 
  score 
}: { 
  label: string; 
  detected: number; 
  ideal: number; 
  score: number;
}) => {
  const color = score > 80 ? "text-green-500" : score > 60 ? "text-yellow-500" : "text-red-500";
  
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/30 last:border-0">
      <div className="flex-1">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs text-muted-foreground">
          Detected: {detected.toFixed(3)} | Ideal: {ideal.toFixed(3)}
        </p>
      </div>
      <div className={`text-lg font-bold ${color}`}>
        {score.toFixed(0)}%
      </div>
    </div>
  );
};

const formatLabel = (key: string): string => {
  const labels: Record<string, string> = {
    eye_spacing: "Eye Spacing",
    nose_chin_ratio: "Nose-Chin Ratio",
    face_aspect: "Face Aspect Ratio",
  };
  return labels[key] || key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
};

export default AnalysisDashboard;
