import { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import PatientForm from "./components/PatientForm";
import PredictionResult from "./components/PredictionResult";
import Footer from "./components/Footer";
import type { PatientFormData, PredictionResponse, HealthResponse } from "./types";
import { Activity, Stethoscope, Brain, Shield } from "lucide-react";

const API_URL = import.meta.env.PROD ? "" : "http://localhost:8000";

export default function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => {});
  }, []);

  const handleSubmit = async (data: PatientFormData) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const json: PredictionResponse = await res.json();
      setResult(json);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar health={health} />

      {/* Hero */}
      <header className="gradient-hero text-white py-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 animate-fade-in-up">
            Hospital No-Show Predictor
          </h1>
          <p className="text-lg md:text-xl text-blue-100 max-w-2xl mx-auto animate-fade-in-up">
            Reduce missed appointments with machine learning. Enter patient details to predict the likelihood of a no-show.
          </p>
          <div className="flex justify-center gap-6 mt-8 flex-wrap animate-fade-in-up">
            {[
              { icon: Brain, label: "XGBoost Model", sub: "AUC-ROC 0.75" },
              { icon: Activity, label: "17 Features", sub: "Temporal + Clinical" },
              { icon: Shield, label: "Production Ready", sub: "FastAPI + Docker" },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-3 bg-white/10 rounded-xl px-5 py-3 backdrop-blur-sm">
                <item.icon className="w-6 h-6 text-blue-200" />
                <div className="text-left">
                  <div className="font-semibold text-sm">{item.label}</div>
                  <div className="text-xs text-blue-200">{item.sub}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 py-10 w-full">
        <div className="grid lg:grid-cols-2 gap-8">
          <PatientForm onSubmit={handleSubmit} loading={loading} />
          <PredictionResult result={result} error={error} loading={loading} />
        </div>
      </main>

      <Footer />
    </div>
  );
}
