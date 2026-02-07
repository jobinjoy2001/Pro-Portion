import { RealtimeGridData } from "@/lib/api";
import { motion } from "framer-motion";
import { RotateCcw, Eye, Ruler, Activity } from "lucide-react";

interface LiveInfoPanelProps {
  data: RealtimeGridData | null;
  isConnected: boolean;
  fps: number;
}

const LiveInfoPanel = ({ data, isConnected, fps }: LiveInfoPanelProps) => {
  return (
    <div className="space-y-4">
      {/* Connection status */}
      <div className="glass-card p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-destructive"}`} />
          <span className="text-sm font-medium">
            {isConnected ? "Connected" : "Disconnected"}
          </span>
        </div>
        <span className="mono-text text-muted-foreground">{fps} FPS</span>
      </div>

      {data && (
        <>
          {/* Head Pose */}
          <div className="glass-card p-5 space-y-4">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <RotateCcw className="h-4 w-4 text-primary" />
              Head Pose
            </div>
            <div className="grid grid-cols-3 gap-3">
              <AngleDisplay label="Pitch" value={data.pose.pitch} />
              <AngleDisplay label="Yaw" value={data.pose.yaw} />
              <AngleDisplay label="Roll" value={data.pose.roll} />
            </div>
          </div>

          {/* View Type */}
          <div className="glass-card p-5 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Eye className="h-4 w-4 text-primary" />
              View Classification
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1.5 rounded-xl bg-primary/10 text-primary text-sm font-semibold">
                {data.view_type}
              </span>
              <span className="text-xs text-muted-foreground">
                {getViewDescription(data.view_type)}
              </span>
            </div>
          </div>

          {/* Status */}
          <div className="glass-card p-5 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Activity className="h-4 w-4 text-primary" />
              Status
            </div>
            <p className="text-sm text-muted-foreground">{data.status}</p>
          </div>
        </>
      )}

      {!data && isConnected && (
        <div className="glass-card p-8 flex flex-col items-center justify-center text-center">
          <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-4 animate-pulse-glow">
            <Ruler className="h-6 w-6 text-primary" />
          </div>
          <p className="text-sm text-muted-foreground">
            Position your face in front of the camera to start tracking
          </p>
        </div>
      )}
    </div>
  );
};

const AngleDisplay = ({ label, value }: { label: string; value: number }) => (
  <div className="text-center p-3 rounded-xl bg-secondary/40">
    <p className="mono-text text-lg font-semibold">{value.toFixed(1)}°</p>
    <p className="text-xs text-muted-foreground mt-1">{label}</p>
  </div>
);

function getViewDescription(viewType: string): string {
  const descriptions: Record<string, string> = {
    "Front": "0-12° rotation",
    "Three-Quarter": "12-40° rotation",
    "Profile": "40-75° rotation",
    "Tilted": "Head tilted",
  };
  return descriptions[viewType] || "";
}

export default LiveInfoPanel;
