# -*- coding: utf-8 -*-
"""
@Time    : 2026/2/11 18:29
@Author  : Terry_CYY
@File    : 4_Conflict_Full_Statistics.py
@IDE     : PyCharm
@Function: Integrated statistical pipeline for unified conflict thresholds (TTC <= 2s, PET(DGT) <= 4s).
"""

""" 
Output (Summary printed by dataset):
  1) Total conflict pairs (number of unique vehicle interactions).
  2) Categorical statistics of conflict types (based on heading difference + relative position rules).
  3) Number of involved unique vehicles (deduplicated IDs within the dataset).
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Threshold Configuration
TTC_MIN = 0.0
TTC_MAX = 2.0
PET_MIN = 0.0
PET_MAX = 4.0


def _pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    """Generate a unique key for a vehicle pair regardless of order."""
    a = a.astype(str)
    b = b.astype(str)
    left = np.where(a <= b, a, b)
    right = np.where(a <= b, b, a)
    return pd.Series(left, index=a.index) + "|" + pd.Series(right, index=a.index)


def _scope_from_file_name(file_name: str) -> str:
    """Extract scope/prefix from file name."""
    return os.path.splitext(os.path.basename(file_name))[0]


def _scoped_obj_id(scope: str, scene: pd.Series | None, obj_id: pd.Series) -> pd.Series:
    """Generate a globally unique ID including scope and scene info."""
    obj_id = obj_id.astype(str)
    if scene is None:
        return scope + "_" + obj_id
    return scope + "_" + scene.astype(str) + "_" + obj_id


def _normalize_to_360(angle_deg: pd.Series) -> pd.Series:
    """Normalize angles to the range [0, 360)."""
    return np.mod(angle_deg, 360.0)


def _convert_angle_series_to_degrees(s: pd.Series) -> pd.Series:
    """Detect if angles are in radians and convert to degrees if necessary."""
    s = pd.to_numeric(s, errors="coerce")
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return _normalize_to_360(s)
    q01 = float(finite.quantile(0.01))
    q99 = float(finite.quantile(0.99))
    # Heuristic: if values fall within [-2pi, 2pi], treat as radians
    is_radian = (q01 >= -2 * np.pi - 0.2) and (q99 <= 2 * np.pi + 0.2)
    if is_radian:
        s = np.degrees(np.mod(s, 2 * np.pi))
    return _normalize_to_360(s)


def _add_diff_column(df: pd.DataFrame, angle1_col: str, angle2_col: str) -> pd.DataFrame:
    """Calculate the absolute heading difference (acute angle)."""
    df = df.copy()
    df[angle1_col] = _convert_angle_series_to_degrees(df[angle1_col])
    df[angle2_col] = _convert_angle_series_to_degrees(df[angle2_col])
    diff = np.abs(df[angle1_col] - df[angle2_col])
    df["diff"] = np.where(diff > 180, 360 - diff, diff)
    df["diff"] = pd.to_numeric(df["diff"], errors="coerce").round(2)
    return df


def _classify_collision(x1, y1, x2, y2, angle1, angle2, diff) -> str:
    """Geometric classification of collision types based on heading and relative position."""
    dx = x2 - x1
    dy = y2 - y1
    line_angle = np.degrees(np.arctan2(dy, dx)) % 360

    angle_diff1 = min((line_angle - angle1) % 180, (angle1 - line_angle) % 180)
    angle_diff2 = min((line_angle - angle2) % 180, (angle2 - line_angle) % 180)

    if diff > 120:
        return "Head-on"
    if 30 <= diff <= 120:
        return "Crossing/Angle"
    if diff < 30:
        if angle_diff1 < 30 and angle_diff2 < 30:
            return "Rear-end"
        return "Sideswipe"
    return "Unknown"


def _build_pet_pair_map(pet_df: pd.DataFrame, scope: str) -> tuple[set[str], dict[str, float]]:
    """Filter PET/DGT data and return a mapping of unique pair keys to minimum PET values."""
    pet_df = pet_df.copy()

    id1_col = "id1" if "id1" in pet_df.columns else ("ID1" if "ID1" in pet_df.columns else None)
    id2_col = "id2" if "id2" in pet_df.columns else ("ID2" if "ID2" in pet_df.columns else None)
    if id1_col is None or id2_col is None:
        return set(), {}

    value_col = "PET" if "PET" in pet_df.columns else ("DGT" if "DGT" in pet_df.columns else None)
    if value_col is None:
        return set(), {}

    pet_df[value_col] = pd.to_numeric(pet_df[value_col], errors="coerce")
    pet_df = pet_df.loc[(pet_df[value_col] >= PET_MIN) & (pet_df[value_col] <= PET_MAX)].copy()

    # Remove self-interactions
    pet_df = pet_df.loc[pet_df[id1_col].astype(str) != pet_df[id2_col].astype(str)].copy()
    if pet_df.empty:
        return set(), {}

    has_scene = "Scene" in pet_df.columns

    if not has_scene:
        obj1 = _scoped_obj_id(scope, None, pet_df[id1_col])
        obj2 = _scoped_obj_id(scope, None, pet_df[id2_col])
        pair_key = _pair_key(obj1, obj2)
        pet_df["_pair_key"] = pair_key
        pet_min = pet_df.groupby("_pair_key", dropna=True)[value_col].min()
        keys = set(pet_min.index.astype(str).tolist())
        return keys, pet_min.to_dict()

    scene = pet_df["Scene"]
    obj1 = _scoped_obj_id(scope, scene, pet_df[id1_col])
    obj2 = _scoped_obj_id(scope, scene, pet_df[id2_col])
    pet_df["_pair_key"] = _pair_key(obj1, obj2)
    pet_min = pet_df.groupby("_pair_key", dropna=True)[value_col].min()
    keys = set(pet_min.index.astype(str).tolist())

    return keys, pet_min.to_dict()


@dataclass(frozen=True)
class FileStats:
    dataset: str
    file_name: str
    pair_level: pd.DataFrame


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _select_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _write_result_csv(df: pd.DataFrame, result_dir: str, file_name: str) -> None:
    _ensure_dir(result_dir)
    out_path = os.path.join(result_dir, file_name)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def _compute_file_pair_level(dataset: str, ssm_path: str, pet_path: str) -> FileStats | None:
    """Main logic for joining TTC and PET data and selecting representative conflict events."""
    file_name = os.path.basename(ssm_path)
    scope = _scope_from_file_name(file_name)
    ssm_df = pd.read_csv(ssm_path)
    pet_df = pd.read_csv(pet_path)

    required_cols = {"Car 1 ID", "Car 2 ID", "TTC", "Car 1 X", "Car 1 Y", "Car 2 X", "Car 2 Y", "Car 1 Angle", "Car 2 Angle"}
    missing = required_cols - set(ssm_df.columns)
    if missing:
        print(f"Skipping {file_name}: 2DSSM missing columns {', '.join(sorted(missing))}")
        return None

    pet_pair_keys, pet_value_map = _build_pet_pair_map(pet_df, scope=scope)
    if len(pet_pair_keys) == 0:
        empty = pd.DataFrame(columns=["pair_key", "obj1_id", "obj2_id", "DGT/PET", "conflict_type", "TTC"])
        return FileStats(dataset=dataset, file_name=file_name, pair_level=empty)

    ssm_df["TTC"] = pd.to_numeric(ssm_df["TTC"], errors="coerce")
    filtered = ssm_df.loc[(ssm_df["TTC"] >= TTC_MIN) & (ssm_df["TTC"] <= TTC_MAX)].copy()
    if filtered.empty:
        empty = pd.DataFrame(columns=["pair_key", "obj1_id", "obj2_id", "DGT/PET", "conflict_type", "TTC"])
        return FileStats(dataset=dataset, file_name=file_name, pair_level=empty)

    car1 = filtered["Car 1 ID"].astype(str)
    car2 = filtered["Car 2 ID"].astype(str)

    if "Scene" in filtered.columns:
        scene = filtered["Scene"]
        obj1_id = _scoped_obj_id(scope, scene, car1)
        obj2_id = _scoped_obj_id(scope, scene, car2)
        pair_key = _pair_key(obj1_id, obj2_id)
        pet_mask = pair_key.isin(pet_pair_keys)
    else:
        obj1_id = _scoped_obj_id(scope, None, car1)
        obj2_id = _scoped_obj_id(scope, None, car2)
        pair_key = _pair_key(obj1_id, obj2_id)
        pet_mask = pair_key.isin(pet_pair_keys)

    filtered = filtered.loc[pet_mask].copy()
    if filtered.empty:
        empty = pd.DataFrame(columns=["pair_key", "obj1_id", "obj2_id", "DGT/PET", "conflict_type", "TTC"])
        return FileStats(dataset=dataset, file_name=file_name, pair_level=empty)

    filtered["pair_key"] = pair_key.loc[pet_mask].values
    filtered["obj1_id"] = obj1_id.loc[pet_mask].values
    filtered["obj2_id"] = obj2_id.loc[pet_mask].values
    filtered["DGT/PET"] = filtered["pair_key"].astype(str).map(pet_value_map)

    filtered = _add_diff_column(filtered, "Car 1 Angle", "Car 2 Angle")
    filtered["conflict_type"] = filtered.apply(
        lambda row: _classify_collision(
            row["Car 1 X"],
            row["Car 1 Y"],
            row["Car 2 X"],
            row["Car 2 Y"],
            row["Car 1 Angle"],
            row["Car 2 Angle"],
            row["diff"],
        ),
        axis=1,
    )

    # Filter out redundant "Sideswipe" cases with zero heading difference
    filtered = filtered.loc[~((filtered["diff"] <= 1) & (filtered["conflict_type"] == "Sideswipe"))].copy()
    if filtered.empty:
        empty = pd.DataFrame(columns=["pair_key", "obj1_id", "obj2_id", "DGT/PET", "conflict_type", "TTC"])
        return FileStats(dataset=dataset, file_name=file_name, pair_level=empty)

    def _pick_pair_row(g: pd.DataFrame) -> pd.DataFrame:
        """Select the row with the lowest TTC for each conflict pair."""
        modes = g["conflict_type"].mode()
        mode_type = modes.iat[0] if len(modes) else "Unknown"
        g2 = g.loc[g["conflict_type"] == mode_type]
        return g2.nsmallest(1, "TTC", keep="first")

    pair_level = (
        filtered.groupby("pair_key", group_keys=False)
        .apply(_pick_pair_row)
        .reset_index(drop=True)
    )

    frame_col = "Frame" if "Frame" in pair_level.columns else ("Time" if "Time" in pair_level.columns else None)
    base_cols = [
        "Scene",
        frame_col,
        "Car 1 ID",
        "Car 2 ID",
        "Car 1 X",
        "Car 1 Y",
        "Car 2 X",
        "Car 2 Y",
        "diff",
        "TTC",
        "MTTC",
        "DGT/PET",
        "conflict_type",
    ]
    selected = _select_existing_cols(pair_level, [c for c in base_cols if c is not None])
    if "pair_key" not in selected:
        selected = ["pair_key"] + selected
    for c in ["obj1_id", "obj2_id"]:
        if c in pair_level.columns and c not in selected:
            selected.append(c)
    pair_level_out = pair_level[selected].copy()
    return FileStats(dataset=dataset, file_name=file_name, pair_level=pair_level_out)


def _summarize_pair_level(title: str, pair_level: pd.DataFrame) -> None:
    """Print categorical summary for a specific file or dataset."""
    pair_count = int(pair_level["pair_key"].nunique()) if (not pair_level.empty and "pair_key" in pair_level.columns) else 0
    if pair_level.empty:
        type_counts = pd.Series(dtype=int)
        unique_vehicles = 0
    else:
        type_counts = pair_level["conflict_type"].value_counts() if "conflict_type" in pair_level.columns else pd.Series(dtype=int)
        if "obj1_id" in pair_level.columns and "obj2_id" in pair_level.columns:
            vehicles = pd.Series(pd.concat([pair_level["obj1_id"], pair_level["obj2_id"]]).unique())
            unique_vehicles = int(vehicles.nunique())
        else:
            unique_vehicles = 0

    if len(type_counts) == 0:
        type_text = "(None)"
    else:
        type_text = ", ".join([f"{k}: {int(v)}" for k, v in type_counts.items()])

    print(f"{title} | Pairs: {pair_count} | Involved Vehicles: {unique_vehicles} | Types: {type_text}")


def _summarize_dataset(dataset: str, file_stats: list[FileStats]) -> None:
    """Consolidate file statistics into a dataset-level summary."""
    if not file_stats:
        print(f"No valid data for dataset {dataset}.")
        return

    pair_level = pd.concat([fs.pair_level.assign(file=fs.file_name) for fs in file_stats], ignore_index=True)

    print("=" * 70)
    print(f"Dataset: {dataset}")
    print(f"Thresholds: TTC in [{TTC_MIN}, {TTC_MAX}]s, PET in [{PET_MIN}, {PET_MAX}]s")

    if not pair_level.empty and "file" in pair_level.columns:
        for file_name in pair_level["file"].dropna().astype(str).unique().tolist():
            file_df = pair_level.loc[pair_level["file"].astype(str) == file_name].copy()
            file_title = os.path.splitext(file_name)[0]
            _summarize_pair_level(file_title, file_df)
    else:
        print("(No file data available)")

    _summarize_pair_level("--- DATASET SUMMARY ---", pair_level)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated conflict statistics script.")
    parser.add_argument("--ssm_dir", type=str, default="2DSSM", help="Input directory for 2D-SSM files")
    parser.add_argument("--pet_dir", type=str, default="PET", help="Input directory for PET files")
    parser.add_argument("--result_dir", type=str, default="result", help="Output directory for processed results")
    parser.add_argument("--no_save_result", action="store_true", default=False, help="Do not save CSV result files")
    args = parser.parse_args()

    dataset_groups: list[tuple[str, list[str]]] = [
        (
            "INTERACTION",
            [
                "DR_USA_Intersection_EP0.csv",
                "DR_USA_Intersection_EP1.csv",
                "DR_USA_Intersection_GL.csv",
                "DR_USA_Intersection_MA.csv",
            ],
        ),
        ("inD", ["loc1.csv", "loc2.csv", "loc3.csv", "loc4.csv"]),
        ("CitySim", ["IntersectionA.csv", "IntersectionB.csv", "IntersectionD.csv", "IntersectionE.csv"]),
        ("SinD", ["Tianjin.csv", "Xi'an.csv", "Chongqing.csv", "Changchun.csv"]),
        ("SongdoTraffic", ["SongdoTraffic.csv"]),
        ("FLUID", ["FI.csv", "FIDRT.csv", "TI.csv"]),
    ]

    for dataset, files in dataset_groups:
        stats: list[FileStats] = []
        for fname in files:
            ssm_path = os.path.join(args.ssm_dir, fname)
            pet_path = os.path.join(args.pet_dir, fname)
            if not os.path.exists(ssm_path):
                print(f"Skipping {dataset}/{fname}: {ssm_path} not found.")
                continue
            if not os.path.exists(pet_path):
                print(f"Skipping {dataset}/{fname}: {pet_path} not found.")
                continue
            fs = _compute_file_pair_level(dataset, ssm_path, pet_path)
            if fs is not None:
                stats.append(fs)
                if not args.no_save_result:
                    _write_result_csv(fs.pair_level, args.result_dir, fname)

        _summarize_dataset(dataset, stats)


if __name__ == "__main__":
    main()