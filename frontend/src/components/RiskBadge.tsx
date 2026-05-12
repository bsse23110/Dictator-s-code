import { Shield, AlertTriangle, AlertCircle } from "lucide-react";

interface Props {
  level: string;
}

const config: Record<string, { icon: typeof Shield; color: string; bg: string }> = {
  "Low Risk": { icon: Shield, color: "text-green-700", bg: "bg-green-100 border-green-300" },
  "Medium Risk": { icon: AlertCircle, color: "text-amber-700", bg: "bg-amber-100 border-amber-300" },
  "High Risk": { icon: AlertTriangle, color: "text-red-700", bg: "bg-red-100 border-red-300" },
};

export default function RiskBadge({ level }: Props) {
  const cfg = config[level] || config["Low Risk"];
  const Icon = cfg.icon;

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-sm font-semibold ${cfg.bg} ${cfg.color}`}>
      <Icon className="w-4 h-4" />
      {level}
    </span>
  );
}
