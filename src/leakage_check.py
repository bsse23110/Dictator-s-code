# src/leakage_check.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import config as cfg

def detect_leakage(X, y, corr_threshold=0.90):
    print("\n[Phase 1.3] Running leakage & feature quality checks...")
    warnings = []
    safe_features = list(X.columns)

    # 1. Target leakage / High correlation
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            corr = abs(X[col].corr(y))
            if corr > corr_threshold:
                warnings.append(f"⚠️  High target correlation: {col} (r={corr:.3f})")
                safe_features.remove(col)

    # 2. Zero variance / constant columns
    for col in X.columns:
        if X[col].nunique() <= 1:
            warnings.append(f"⚠️  Zero variance: {col}")
            safe_features.remove(col)

    # 3. Temporal / Forward-looking keywords
    future_kw = ["future", "next", "remaining", "post_", "upcoming", "tomorrow"]
    for col in X.columns:
        if any(kw in col.lower() for kw in future_kw):
            warnings.append(f"⚠️  Potential temporal leak: {col}")
            safe_features.remove(col)

    # 4. Target encoding sanity (neighbourhood_no_show_rate)
    if "neighbourhood_no_show_rate" in X.columns:
        corr = abs(X["neighbourhood_no_show_rate"].corr(y))
        if corr > 0.98:
            warnings.append("⚠️  Target encoding may be overfit: neighbourhood_no_show_rate")

    if warnings:
        for w in warnings: print(f"  {w}")
    else:
        print("  ✅ No obvious leakage detected.")

    print(f"  📦 Features retained: {len(safe_features)}/{X.shape[1]}")
    return safe_features

if __name__ == "__main__":
    df = pd.read_csv(cfg.PREPROCESSED_PATH)
    detect_leakage(df.drop(columns=[cfg.TARGET_COL]), df[cfg.TARGET_COL])
