import { useState } from "react";
import type { PatientFormData } from "@/types";
import { User, Calendar, MapPin, Clock, Phone, Loader2, Send } from "lucide-react";

interface PatientFormProps {
  onSubmit: (data: PatientFormData) => void;
  loading: boolean;
}

const NEIGHBOURHOODS = [
  "AEROPORTO", "ANDORINHAS", "ANTÔNIO HONÓRIO", "ARIOVALDO FAVALESSA",
  "BARRO VERMELHO", "BELA VISTA", "BENTO FERREIRA", "BOA VISTA",
  "BONFIM", "CARATOÍRA", "CENTRO", "COMDUSA", "CONQUISTA",
  "CONSOLAÇÃO", "CRUZAMENTO", "DA PENHA", "DE LOURDES",
  "DO CABRAL", "DO MOSCOSO", "DO QUADRO", "ENSEADA DO SUÁ",
  "ESTRELINHA", "FONTE GRANDE", "FORTE SÃO JOÃO", "FRADINHOS",
  "GOIABEIRAS", "GRANDE VITÓRIA", "GURIGICA", "HORTO",
  "ILHA DAS CAIEIRAS", "ILHA DE SANTA MARIA", "ILHA DO BOI",
  "ILHA DO FRADE", "ILHA DO PRÍNCIPE", "ILHAS OCEÂNICAS DE TRINDADE",
  "INHANGUETÁ", "ITARARÉ", "JABOUR", "JARDIM CAMBURI",
  "JARDIM DA PENHA", "JESUS DE NAZARETH", "JOANA D'ARC", "JUCUTUQUARA",
  "MARIA ORTIZ", "MARUÍPE", "MATA DA PRAIA", "MONTE BELO",
  "MORADA DE CAMBURI", "MÁRIO CYPRESTE", "NAZARETH", "NOVA PALESTINA",
  "PARQUE MOSCOSO", "PARQUE RESIDENCIAL LARANJEIRAS", "PENHA", "PIEDADE",
  "PONTAL DE CAMBURI", "PRAIA DO CANTO", "PRAIA DO SUÁ", "REDENÇÃO",
  "REPÚBLICA", "RESISTÊNCIA", "ROMÃO", "SANTA CECÍLIA",
  "SANTA CLARA", "SANTA HELENA", "SANTA LUÍZA", "SANTA LÚCIA",
  "SANTA MARTHA", "SANTA TEREZA", "SANTO ANDRÉ", "SANTO ANTÔNIO",
  "SANTOS DUMONT", "SEGURANÇA DO LAR", "SOLON BORGES", "SÃO BENEDITO",
  "SÃO CRISTÓVÃO", "SÃO JOSÉ", "SÃO PEDRO", "TABUAZEIRO",
  "UNIVERSITÁRIO", "VILA RUBIM",
];

export default function PatientForm({ onSubmit, loading }: PatientFormProps) {
  const [form, setForm] = useState<PatientFormData>({
    gender: 0,
    age: 35,
    scholarship: 0,
    hypertension: 0,
    diabetes: 0,
    alcoholism: 0,
    handicap: 0,
    sms_received: 1,
    scheduled_date: new Date().toISOString().split("T")[0],
    appointment_date: new Date(Date.now() + 7 * 86400000).toISOString().split("T")[0],
    neighbourhood: "JARDIM DA PENHA",
    previous_appointments: 0,
    previous_no_shows: 0,
  });

  const update = (key: keyof PatientFormData, value: any) =>
    setForm((f) => ({ ...f, [key]: value }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(form);
  };

  const toggle = (key: keyof PatientFormData) =>
    update(key, form[key] ? 0 : 1);

  const btnClass = (active: boolean) =>
    `px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
      active
        ? "bg-primary text-white shadow-md shadow-primary/25"
        : "bg-secondary text-secondary-foreground hover:bg-secondary/70"
    }`;

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-lg border p-6 space-y-5">
      <h2 className="text-xl font-bold flex items-center gap-2">
        <User className="w-5 h-5 text-primary" />
        Patient Information
      </h2>

      {/* Gender */}
      <div>
        <label className="text-sm font-medium text-muted-foreground mb-1.5 block">Gender</label>
        <div className="flex gap-2">
          <button type="button" onClick={() => update("gender", 0)} className={btnClass(form.gender === 0)}>Male</button>
          <button type="button" onClick={() => update("gender", 1)} className={btnClass(form.gender === 1)}>Female</button>
        </div>
      </div>

      {/* Age */}
      <div>
        <label className="text-sm font-medium text-muted-foreground mb-1.5 block">Age: <span className="font-bold text-foreground">{form.age}</span></label>
        <input
          type="range" min={0} max={100} value={form.age}
          onChange={(e) => update("age", parseInt(e.target.value))}
          className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground mt-1"><span>0</span><span>100</span></div>
      </div>

      {/* Medical Conditions Grid */}
      <div>
        <label className="text-sm font-medium text-muted-foreground mb-2 block">Medical Conditions</label>
        <div className="grid grid-cols-2 gap-2">
          {[
            ["scholarship", "Scholarship (Bolsa Família)"],
            ["hypertension", "Hypertension"],
            ["diabetes", "Diabetes"],
            ["alcoholism", "Alcoholism"],
            ["handicap", "Handicap"],
            ["sms_received", "SMS Reminder Received"],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => toggle(key as keyof PatientFormData)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border transition-all ${
                form[key as keyof PatientFormData]
                  ? "bg-primary/10 border-primary text-primary"
                  : "bg-secondary border-transparent text-muted-foreground hover:border-border"
              }`}
            >
              <span className={`w-4 h-4 rounded border-2 flex items-center justify-center text-[10px] transition-all ${
                form[key as keyof PatientFormData] ? "bg-primary border-primary text-white" : "border-muted-foreground/30"
              }`}>
                {form[key as keyof PatientFormData] ? "✓" : ""}
              </span>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Dates */}
      <div className="grid sm:grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium text-muted-foreground mb-1.5 flex items-center gap-1.5">
            <Calendar className="w-4 h-4" /> Scheduled Date
          </label>
          <input
            type="date" value={form.scheduled_date}
            onChange={(e) => update("scheduled_date", e.target.value)}
            className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-muted-foreground mb-1.5 flex items-center gap-1.5">
            <Calendar className="w-4 h-4" /> Appointment Date
          </label>
          <input
            type="date" value={form.appointment_date}
            onChange={(e) => update("appointment_date", e.target.value)}
            className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
          />
        </div>
      </div>

      {/* Neighbourhood */}
      <div>
        <label className="text-sm font-medium text-muted-foreground mb-1.5 flex items-center gap-1.5">
          <MapPin className="w-4 h-4" /> Neighbourhood
        </label>
        <select
          value={form.neighbourhood}
          onChange={(e) => update("neighbourhood", e.target.value)}
          className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all bg-white"
        >
          {NEIGHBOURHOODS.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
      </div>

      {/* Previous History */}
      <div className="grid sm:grid-cols-2 gap-4">
        <div>
          <label className="text-sm font-medium text-muted-foreground mb-1.5 flex items-center gap-1.5">
            <Clock className="w-4 h-4" /> Previous Appointments
          </label>
          <input
            type="number" min={0} value={form.previous_appointments}
            onChange={(e) => update("previous_appointments", parseInt(e.target.value) || 0)}
            className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-muted-foreground mb-1.5 flex items-center gap-1.5">
            <Phone className="w-4 h-4" /> Previous No-Shows
          </label>
          <input
            type="number" min={0} max={form.previous_appointments} value={form.previous_no_shows}
            onChange={(e) => update("previous_no_shows", parseInt(e.target.value) || 0)}
            className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
          />
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 bg-primary text-white rounded-xl font-semibold hover:bg-primary/90 active:scale-[0.98] transition-all flex items-center justify-center gap-2 disabled:opacity-60 shadow-lg shadow-primary/25"
      >
        {loading ? (
          <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing...</>
        ) : (
          <><Send className="w-5 h-5" /> Predict No-Show Risk</>
        )}
      </button>
    </form>
  );
}
