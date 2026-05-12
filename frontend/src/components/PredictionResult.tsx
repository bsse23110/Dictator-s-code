import type { PredictionResponse } from "@/types";
import RiskBadge from "./RiskBadge";
import ProbabilityGauge from "./ProbabilityGauge";
import { AlertTriangle, CheckCircle, Loader2, BarChart3 } from "lucide-react";

interface Props {
  result: PredictionResponse | null;
  error: string | null;
  loading: boolean;
}

export default function PredictionResult({ result, error, loading }: Props) {
  return (
    <div className="bg-white rounded-2xl shadow-lg border p-6 min-h-[500px] flex flex-col">
      <h2 className="text-xl font-bold flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-primary" />
        Prediction Result
      </h2>

      {loading && (
        <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground gap-3">
          <Loader2 className="w-12 h-12 animate-spin text-primary" />
          <p className="font-medium">Analyzing patient data...</p>
        </div>
      )}

      {error && (
        <div className="flex-1 flex flex-col items-center justify-center gap-3">
          <AlertTriangle className="w-12 h-12 text-destructive" />
          <p className="text-destructive font-medium">{error}</p>
        </div>
      )}

      {!loading && !error && !result && (
        <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground gap-3">
          <div className="w-16 h-16 rounded-2xl bg-secondary flex items-center justify-center">
            <BarChart3 className="w-8 h-8" />
          </div>
          <p className="font-medium">Fill in the form and click "Predict"</p>
          <p className="text-sm">The AI model will analyze 17 features to estimate no-show risk.</p>
        </div>
      )}

      {result && (
        <div className="flex-1 flex flex-col animate-fade-in-up">
          {/* Risk Badge */}
          <div className="flex items-center justify-between mb-6">
            <RiskBadge level={result.risk_level} />
            <div className="text-right">
              <div className="text-2xl font-bold text-foreground">{(result.probability * 100).toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground">No-Show Probability</div>
            </div>
          </div>

          {/* Gauge */}
          <ProbabilityGauge value={result.probability} riskLevel={result.risk_level} />

          {/* Explanation */}
          <div className={`mt-6 p-4 rounded-xl border ${
            result.risk_level === "Low Risk"
              ? "bg-green-50 border-green-200"
              : result.risk_level === "Medium Risk"
              ? "bg-amber-50 border-amber-200"
              : "bg-red-50 border-red-200"
          }`}>
            <div className="flex items-start gap-2">
              <CheckCircle className={`w-5 h-5 mt-0.5 flex-shrink-0 ${
                result.risk_level === "Low Risk" ? "text-green-600" :
                result.risk_level === "Medium Risk" ? "text-amber-600" : "text-red-600"
              }`} />
              <div>
                <p className="font-semibold text-sm mb-1">
                  {result.risk_level === "Low Risk" ? "Good News" :
                   result.risk_level === "Medium Risk" ? "Caution Advised" : "Action Required"}
                </p>
                <p className="text-sm text-muted-foreground">{result.explanation}</p>
              </div>
            </div>
          </div>

          {/* Feature Quick View */}
          <details className="mt-4 group">
            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground transition-colors">
              View engineered features →
            </summary>
            <div className="mt-2 p-3 bg-secondary rounded-lg max-h-40 overflow-y-auto text-xs font-mono">
              {Object.entries(result.features_used).map(([key, val]) => (
                <div key={key} className="flex justify-between py-0.5">
                  <span className="text-muted-foreground">{key}</span>
                  <span className="font-medium">{String(val)}</span>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
