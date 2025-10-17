from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- helpers ----------
def _read_mapping_txt(txt_path: Path) -> pd.DataFrame:
    """
    Učita linije tipa:
      '0039552 Boeing_737-800'  ili
      '0039552.jpg Boeing_737-800'  ili
      'images/0039552 Boeing_737-800'  ili
      'images/0039552.jpg Boeing_737-800'
    Vrati DataFrame[image, label] gde je 'image' originalni levi token.
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img, label = line.split(" ", 1)
            rows.append((img, label))
    return pd.DataFrame(rows, columns=["image", "label"])

def _normalize_image_name(x: str) -> str:
    """
    Normalizuj levi token iz map fajla na oblik '0039552.jpg'.
    Skini eventualni 'images/' prefiks i dodaj '.jpg' ako fali.
    """
    x = x.strip().replace("\\", "/")
    if x.lower().startswith("images/"):
        x = x[7:]  # ukloni 'images/'
    if not x.lower().endswith(".jpg"):
        x = x + ".jpg"
    return x

def _first_existing(*candidates: Path) -> Path:
    """Vrati prvi postojeći fajl iz liste kandidata (variant ili plain)."""
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Nisam našla mapping fajlove (train/val/test). Proveri data/ folder.")

# ---------- public API ----------
def build_df_from_fgvc(base_dir: str | Path, use_all_splits: bool = True) -> pd.DataFrame:
    """
    Kreira jedinstveni DF sa apsolutnim putanjama do slika i labelama.
    Ako use_all_splits=True, spaja train/val/test mape pa radi sopstvenu podelu.
    """
    base_dir = Path(base_dir)
    data_dir = base_dir / "data"
    images_dir = data_dir / "images"

    # podrži i *_variant_ i plain mape
    train_map = _read_mapping_txt(_first_existing(
        data_dir / "images_variant_train.txt",
        data_dir / "images_train.txt"
    ))
    val_map = _read_mapping_txt(_first_existing(
        data_dir / "images_variant_val.txt",
        data_dir / "images_val.txt"
    ))
    test_map = _read_mapping_txt(_first_existing(
        data_dir / "images_variant_test.txt",
        data_dir / "images_test.txt"
    ))

    if use_all_splits:
        df = pd.concat([train_map, val_map, test_map], ignore_index=True)
    else:
        df = pd.concat([train_map, val_map], ignore_index=True)

    # napravi pune putanje (robustno)
    def _to_full_path(x: str) -> str:
        fname = _normalize_image_name(x)
        return str((images_dir / fname).resolve())

    df["filepath"] = df["image"].apply(_to_full_path)
    df = df[df["filepath"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    return df

def pick_top_k_classes(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    topk = df["label"].value_counts().nlargest(k).index.tolist()
    return df[df["label"].isin(topk)].reset_index(drop=True)

def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Train+Val+Test moraju dati 1.0"
    from sklearn.model_selection import train_test_split as _split

    df_trainval, df_test = _split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_ratio_in_trainval = val_size / (train_size + val_size)
    df_train, df_val = _split(
        df_trainval, test_size=val_ratio_in_trainval, stratify=df_trainval["label"], random_state=seed
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
