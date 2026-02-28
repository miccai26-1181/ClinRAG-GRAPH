
import os
import re
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resolve_npy_by_substring(npy_dir: str, mri_id: str):
    pattern = os.path.join(npy_dir, f"*{mri_id}*.npy")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        return None
    return matches[0]

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_subtype_to_index(s: Any) -> int:
    """
    Map subtype to {0,1,2,3}:
      0: TripleNeg
      1: HER2pos
      2: LuminalA
      3: LuminalB

    Accepts common variants:
      - 'TripleNeg', 'TripleNegative', 'TN'
      - 'HER2pos', 'HER2+', 'HER2 positive'
      - 'Luminal A', 'LumA'
      - 'Luminal B', 'LumB'
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0
    t = str(s).strip().lower()
    t = re.sub(r"[\s_\-]+", "", t)

    if t in {"triplenegative", "tripleneg", "tn", "triple-"} or ("triple" in t and "neg" in t):
        return 0
    if t in {"her2pos", "her2positive", "her2+", "her2"} or ("her2" in t and ("pos" in t or "positive" in t or "+" in t)):
        return 1
    if t in {"luminala", "luma", "lumina"} or ("luminal" in t and t.endswith("a")):
        return 2
    if t in {"luminalb", "lumb", "luminb"} or ("luminal" in t and t.endswith("b")):
        return 3

    return 0

def load_npy_volume(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W], got shape={arr.shape} @ {path}")
    if arr.shape != (192, 256, 256):
        raise ValueError(f"Expected shape (192,256,256), got {arr.shape} @ {path}")
    return arr.astype(np.float32, copy=False)


# ============================================================
#                        Center config
# ============================================================

@dataclass(frozen=True)
class CenterConfig:
    """
    Per-center configuration.
    - name: center name (string key)
    - csv_path: path to that center's csv
    - npy_dir: directory containing .npy volumes named {mri_id}.npy
    """
    name: str
    csv_path: str
    npy_dir: str


def _coerce_center_configs(
    centers: Union[Sequence[CenterConfig], Dict[str, Dict[str, str]]]
) -> List[CenterConfig]:
    """
    Accept either:
      - list/tuple of CenterConfig
      - dict mapping center_name -> {"csv_path": ..., "npy_dir": ...}
    """
    if isinstance(centers, dict):
        out = []
        for name, cfg in centers.items():
            if "csv_path" not in cfg or "npy_dir" not in cfg:
                raise KeyError(f"Center '{name}' must provide keys: csv_path, npy_dir")
            out.append(CenterConfig(name=str(name), csv_path=str(cfg["csv_path"]), npy_dir=str(cfg["npy_dir"])))
        return out
    return list(centers)


# ============================================================
#                          Dataset
# ============================================================

class BreastDCENPYMultiCenterDataset(Dataset):

    def __init__(
        self,
        centers: Union[Sequence[CenterConfig], Dict[str, Dict[str, str]]],
        *,
        normalize: bool = True,
        return_pid: bool = True,
        strict: bool = True,
        center_id_map: Optional[Dict[str, int]] = None,
        id_col_candidates: Sequence[str] = ("mri_id", "pid"),
        pcr_col_candidates: Sequence[str] = ("pcr", "pCR"),
    ):
        super().__init__()
        self.centers = _coerce_center_configs(centers)
        self.normalize = normalize
        self.return_pid = return_pid
        self.strict = strict

        if len(self.centers) == 0:
            raise ValueError("centers is empty")

        if center_id_map is None:
            names = sorted([c.name for c in self.centers])
            self.center_id_map = {n: i for i, n in enumerate(names)}
        else:
            self.center_id_map = {str(k): int(v) for k, v in center_id_map.items()}
            missing = [c.name for c in self.centers if c.name not in self.center_id_map]
            if missing:
                raise KeyError(
                    f"center_id_map missing centers={missing}. "
                    f"Available keys={list(self.center_id_map.keys())}"
                )

        # build records
        self.records: List[Dict[str, Any]] = []
        skipped_missing_npy = 0
        skipped_bad_row = 0

        for c in self.centers:
            if not os.path.isdir(c.npy_dir):
                raise FileNotFoundError(f"npy_dir not found for center={c.name}: {c.npy_dir}")
            if not os.path.isfile(c.csv_path):
                raise FileNotFoundError(f"csv_path not found for center={c.name}: {c.csv_path}")

            df = pd.read_csv(c.csv_path)

            # find id/pcr columns
            id_col = next((k for k in id_col_candidates if k in df.columns), None)
            if id_col is None:
                raise KeyError(f"[{c.name}] CSV must contain one of {id_col_candidates}, got cols={list(df.columns)}")
            pcr_col = next((k for k in pcr_col_candidates if k in df.columns), None)
            if pcr_col is None:
                raise KeyError(f"[{c.name}] CSV must contain one of {pcr_col_candidates}, got cols={list(df.columns)}")

            needed = [id_col, "age", "ER", "PR", "HER2", "Ki67", "subtype", pcr_col]
            for k in needed:
                if k not in df.columns:
                    raise KeyError(f"[{c.name}] CSV missing column '{k}'")

            for _, row in df.iterrows():
                try:
                    mri_id = str(row[id_col]).strip()
                    if mri_id == "" or mri_id.lower() == "nan":
                        skipped_bad_row += 1
                        continue

                    npy_path = _resolve_npy_by_substring(c.npy_dir, mri_id)
                    if npy_path is None:
                        skipped_missing_npy += 1
                        continue  
                    age = _safe_float(row["age"], 0.0)

                    rec = {
                        "pid": mri_id,
                        "center": c.name,
                        "center_id": int(self.center_id_map[c.name]),
                        "npy_path": npy_path,
                        "age": age,
                        "ER": _safe_int(row["ER"], 0),
                        "PR": _safe_int(row["PR"], 0),
                        "HER2": _safe_int(row["HER2"], 0),
                        "Ki67": _safe_float(row["Ki67"], 0.0),
                        "subtype": normalize_subtype_to_index(row["subtype"]),
                        "pcr": _safe_float(row[pcr_col], 0.0),
                    }
                    self.records.append(rec)
                except Exception:
                    skipped_bad_row += 1
                    if self.strict:
                        raise

        if len(self.records) == 0:
            raise RuntimeError("No valid records after center-local matching. Check csv/npy paths and mri_id naming.")

        print(
            f"[Dataset] centers={len(self.centers)} total_n={len(self.records)} "
            f"skipped_missing_npy={skipped_missing_npy} skipped_bad_row={skipped_bad_row} normalize={self.normalize}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        rec = self.records[idx]
        try:
            arr = load_npy_volume(rec["npy_path"])  # [D,H,W]
            x = torch.from_numpy(arr).float()       # [192,256,256]

            tab = {
                "age": torch.tensor(rec["age"], dtype=torch.float32),
                "ER": torch.tensor(rec["ER"], dtype=torch.float32),
                "PR": torch.tensor(rec["PR"], dtype=torch.float32),
                "HER2": torch.tensor(rec["HER2"], dtype=torch.float32),
                "Ki67": torch.tensor(rec["Ki67"], dtype=torch.float32),
                "subtype": torch.tensor(rec["subtype"], dtype=torch.long),
            }
            y = torch.tensor(rec["pcr"], dtype=torch.float32)
            center_id = torch.tensor(rec["center_id"], dtype=torch.long)

            out = {"x_dce": x, "tab": tab, "y": y, "center_id": center_id}
            if self.return_pid:
                out["pid"] = rec["pid"]
                out["dataset"] = rec["center"]
                out["npy_path"] = rec["npy_path"]
            return out
        except Exception:
            if self.strict:
                raise
            return None
