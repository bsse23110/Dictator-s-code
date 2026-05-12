import { Stethoscope, Activity } from "lucide-react";
import type { HealthResponse } from "@/types";

interface NavbarProps {
  health: HealthResponse | null;
}

export default function Navbar({ health }: NavbarProps) {
  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b shadow-sm">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-primary rounded-lg p-1.5">
            <Stethoscope className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg tracking-tight">No-Show Predictor</span>
            <span className="text-xs text-muted-foreground ml-2 hidden sm:inline">ML-Powered</span>
          </div>
        </div>
        {health && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="w-4 h-4 text-green-500" />
            <span>{health.model}</span>
          </div>
        )}
      </div>
    </nav>
  );
}
