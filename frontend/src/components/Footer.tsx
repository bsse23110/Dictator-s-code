export default function Footer() {
  return (
    <footer className="border-t bg-white mt-10">
      <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-sm text-muted-foreground">
        <p>Hospital No-Show Predictor — University ML Project</p>
        <p>Powered by XGBoost · FastAPI · React · Docker</p>
      </div>
    </footer>
  );
}
