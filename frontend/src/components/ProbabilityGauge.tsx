interface Props {
  value: number; // 0 to 1
  riskLevel: string;
}

function colorForRisk(level: string): string {
  if (level === "Low Risk") return "#22c55e";
  if (level === "Medium Risk") return "#f59e0b";
  return "#ef4444";
}

export default function ProbabilityGauge({ value, riskLevel }: Props) {
  const pct = Math.round(value * 100);
  const color = colorForRisk(riskLevel);
  const circumference = 2 * Math.PI * 52;
  const offset = circumference - (value * circumference);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-36 h-36">
        {/* Background circle */}
        <svg className="w-36 h-36 -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" fill="none" stroke="hsl(214, 32%, 91%)" strokeWidth="12" />
          <circle
            cx="60" cy="60" r="52" fill="none"
            stroke={color} strokeWidth="12" strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 1s ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-3xl font-extrabold" style={{ color }}>{pct}%</span>
        </div>
      </div>
      <div className="flex w-full justify-between text-xs text-muted-foreground mt-1 max-w-[180px]">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
