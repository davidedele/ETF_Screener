from pathlib import Path
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_csv(df: pd.DataFrame, out_dir: str | Path, filename: str) -> Path:
    out_path = ensure_dir(out_dir) / filename
    df.to_csv(out_path, index=True)
    return out_path

