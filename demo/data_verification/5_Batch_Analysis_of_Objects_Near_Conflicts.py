# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/16 13:15
@Author  : Terry_CYY
@File    : 5_Batch_Analysis_of_Objects_Near_Conflicts.py
@IDE     : PyCharm
@Function: 
Combined Version: Statistics of objects near conflicts (Efficient + Readable)

Single Batch Process:
1) "All Surrounding Objects" Criteria
   - For a given frame, count all objects (excluding the conflict pair itself) within distance R 
     from either vehicle in the conflict pair.
   - Output columns: surrounding_count_{R} (Optional: surrounding_{type}_{R})

2) "Conflict-Related Objects" Criteria
   - Construct "Conflict Partnerships" from the conflict table: identify which IDs are conflict partners for each vehicle.
   - For a given frame, count objects within distance R that belong to the vehicle's conflict partner set.
   - Output columns: Car 1 conflictNum_{R} / Car 2 conflictNum_{R} / conflictNum_{R}

Usage:
- Select/Modify dataset column names and paths in the "Dataset Configuration Block" at the bottom, then run.

Notes:
- Conflict pairs are read from `result/*.csv`. Rows usually represent unique conflict events (filtered/deduplicated).
- To calculate "Nearby Objects," the Scene must match a trajectory file, and the Frame must exist in that trajectory.
- The script prints: total result rows, unique pair_key count, Scene matched rows, and Frame matched rows. 
- Frame matched rows are used as the denominator for calculating mean values.
"""

import glob
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def build_partners(conflict_df: pd.DataFrame, id1_col: str, id2_col: str) -> Dict[str, Set[str]]:
    """Build 'Conflict Partnerships': Maps each vehicle ID to its set of conflict partners."""
    partners: Dict[str, Set[str]] = {}
    id1 = conflict_df[id1_col].astype(str).to_numpy()
    id2 = conflict_df[id2_col].astype(str).to_numpy()
    for a, b in zip(id1, id2):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def _scene_from_traj_path(traj_path: str) -> str:
    base = os.path.basename(traj_path)
    if base.lower().endswith(".csv"):
        return base[:-4]
    return os.path.splitext(base)[0]


def _norm_key(s: object) -> str:
    x = str(s).strip().lower()
    x = x.replace("’", "'").replace("`", "'").replace("'", "")
    x = x.replace("-", "_").replace(" ", "")
    x = re.sub(r"__+", "_", x)
    return x


def _scene_key_candidates(scene: object) -> List[str]:
    s = str(scene).strip()
    out: List[str] = []

    def add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    add(s)
    add(_norm_key(s))

    if s.endswith("_Veh_smoothed_tracks"):
        add(s[:-len("_Veh_smoothed_tracks")] + "_tracks")
    if s.endswith("_veh_smoothed_tracks"):
        add(s[:-len("_veh_smoothed_tracks")] + "_tracks")

    s2 = s.replace("Veh_smoothed_", "").replace("veh_smoothed_", "")
    add(s2)
    add(_norm_key(s2))

    if s.endswith("_tracks"):
        add(s[:-len("_tracks")])
        add(_norm_key(s[:-len("_tracks")]))

    return out


def _split_scene_prefix_case(scene: object, sep: str = "_", maxsplit: int = 1) -> Tuple[str, Optional[str]]:
    s = str(scene).strip()
    if not s:
        return "", None
    if sep and sep in s:
        parts = s.split(sep, int(maxsplit))
        if len(parts) >= 2:
            prefix = parts[0].strip()
            rest = parts[1].strip()
            return prefix, (rest if rest else None)
    return s, None


def _stable_unique_paths(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for p in paths:
        k = os.path.normcase(os.path.normpath(str(p)))
        if k in seen:
            continue
        seen.add(k)
        out.append(str(p))
    return out


def _candidate_paths_for_scene(scene: object, scene_to_paths: Dict[str, List[str]]) -> List[str]:
    for k in _scene_key_candidates(scene):
        paths = scene_to_paths.get(k) or scene_to_paths.get(_norm_key(k))
        if paths:
            return paths
    return []


def _rank_paths_by_prefer_dirs(paths: Sequence[str], prefer_dirs: Sequence[str]) -> List[str]:
    if not prefer_dirs:
        return list(paths)
    prefers = [os.path.normcase(str(d)) for d in prefer_dirs]

    def score(p: str) -> int:
        pp = os.path.normcase(str(p))
        for i, d in enumerate(prefers):
            if d and d in pp:
                return i
        return 10**9

    return sorted(list(paths), key=lambda p: (score(p),))


def _frame_coverage_count(traj_path: str, traj_frame_col: str, target_frames: Set[int]) -> int:
    if not target_frames:
        return 0
    remaining = set(target_frames)
    try:
        for chunk in pd.read_csv(traj_path, usecols=[traj_frame_col], chunksize=200000):
            s = pd.to_numeric(chunk[traj_frame_col], errors="coerce").dropna().astype(int)
            if s.empty:
                continue
            hit = set(s.unique().tolist()) & remaining
            if hit:
                remaining -= hit
                if not remaining:
                    break
    except Exception:
        return 0
    return len(target_frames) - len(remaining)


def _extract_date_prefix(scene: object) -> Optional[str]:
    s = str(scene)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    head = s.split("_", 1)[0]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", head):
        return head
    return None


def _index_date_files(traj_globs: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pattern in traj_globs:
        for p in glob.glob(pattern, recursive=True):
            if not p.lower().endswith(".csv"):
                continue
            key = os.path.splitext(os.path.basename(p))[0]
            out.setdefault(key, p)
    return out


def _ms_from_seconds(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    return (s * 1000.0).round().astype("Int64")


def _time_ms_match_mask(values_ms: np.ndarray, targets_sorted_ms: np.ndarray, tol_ms: int) -> np.ndarray:
    if values_ms.size == 0 or targets_sorted_ms.size == 0:
        return np.zeros(values_ms.shape[0], dtype=bool)
    idx = np.searchsorted(targets_sorted_ms, values_ms, side="left")
    idx0 = np.clip(idx - 1, 0, targets_sorted_ms.size - 1)
    idx1 = np.clip(idx, 0, targets_sorted_ms.size - 1)
    d0 = np.abs(targets_sorted_ms[idx0] - values_ms)
    d1 = np.abs(targets_sorted_ms[idx1] - values_ms)
    return np.minimum(d0, d1) <= int(tol_ms)


def _load_songdo_traj_selected(
    traj_path: str,
    traj_scene_col: str,
    traj_time_col: str,
    usecols: List[str],
    scene_to_targets_sorted_ms: Dict[str, np.ndarray],
    tol_ms: int,
    chunksize: int,
) -> pd.DataFrame:
    scenes = set(scene_to_targets_sorted_ms.keys())
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(traj_path, usecols=usecols, chunksize=int(chunksize)):
        scene_ser = chunk[traj_scene_col].astype(str)
        m_scene = scene_ser.isin(scenes)
        if not bool(m_scene.any()):
            continue
        chunk = chunk.loc[m_scene].copy()
        chunk[traj_scene_col] = chunk[traj_scene_col].astype(str)
        ms = _ms_from_seconds(chunk[traj_time_col]).to_numpy(dtype=np.int64, na_value=-1)
        chunk["__frame_ms"] = ms

        kept: List[pd.DataFrame] = []
        for sc in chunk[traj_scene_col].unique().tolist():
            t = scene_to_targets_sorted_ms.get(sc)
            if t is None or t.size == 0:
                continue
            sub = chunk[chunk[traj_scene_col] == sc]
            mask = _time_ms_match_mask(sub["__frame_ms"].to_numpy(dtype=np.int64, copy=False), t, tol_ms)
            if bool(mask.any()):
                kept.append(sub.loc[mask])
        if kept:
            parts.append(pd.concat(kept, ignore_index=True))
    if not parts:
        return pd.DataFrame(columns=usecols + ["__frame_ms"])
    return pd.concat(parts, ignore_index=True)


def index_traj_files(traj_globs: Sequence[str]) -> Dict[str, List[str]]:
    scene_to_paths: Dict[str, List[str]] = {}
    for pattern in traj_globs:
        for p in glob.glob(pattern, recursive=True):
            if not p.lower().endswith(".csv"):
                continue
            scene = _scene_from_traj_path(p)
            scene_to_paths.setdefault(scene, []).append(p)
            scene_to_paths.setdefault(_norm_key(scene), []).append(p)
            if scene.lower().endswith("_tracks"):
                base = scene[: -len("_tracks")]
                scene_to_paths.setdefault(base, []).append(p)
                scene_to_paths.setdefault(_norm_key(base), []).append(p)
    for scene, paths in scene_to_paths.items():
        scene_to_paths[scene] = _stable_unique_paths(paths)
    return scene_to_paths


def load_result_conflicts(result_csv: str, required_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(result_csv)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Result file missing required columns: {missing} | file={result_csv}")
    return df


def _select_one_traj_path(scene: str, scene_to_paths: Dict[str, List[str]]) -> Optional[str]:
    for k in _scene_key_candidates(scene):
        paths = scene_to_paths.get(k) or scene_to_paths.get(_norm_key(k))
        if not paths:
            continue
        if len(paths) == 1:
            return paths[0]
        print(f"Warning: Scene={scene} matched multiple trajectory files, defaulting to the first one: {paths[0]}")
        return paths[0]
    return None


def _select_traj_path_by_result_tag(file_tag: str, scene_to_paths: Dict[str, List[str]]) -> Optional[str]:
    paths = scene_to_paths.get(file_tag) or scene_to_paths.get(_norm_key(file_tag))
    if paths:
        if len(paths) == 1:
            return paths[0]
        print(f"Warning: result={file_tag} matched multiple trajectory files, defaulting to the first one: {paths[0]}")
        return paths[0]

    nt = _norm_key(file_tag)
    cand_paths: List[str] = []
    for k, ps in scene_to_paths.items():
        nk = _norm_key(k)
        if nt and (nt in nk or nk in nt):
            cand_paths.extend(ps)
    cand_paths = _stable_unique_paths(cand_paths)
    if not cand_paths:
        return None
    if len(cand_paths) == 1:
        return cand_paths[0]
    print(f"Warning: result={file_tag} ambiguously matched multiple trajectory files, defaulting to the first: {cand_paths[0]}")
    return cand_paths[0]


def _init_type_columns(conflict_df: pd.DataFrame, traj_df: pd.DataFrame, type_col: str, radii: Sequence[int]) -> List[str]:
    """Initialize columns for specific object type statistics (Optional)."""
    if not type_col or type_col not in traj_df.columns:
        return []
    type_values = pd.Index(traj_df[type_col].astype(str).dropna().unique()).tolist()
    type_values = [t for t in type_values if t != ""]
    out_cols: List[str] = []
    for r in radii:
        for t in type_values:
            col = f"surrounding_{t}_{r}"
            if col not in conflict_df.columns:
                conflict_df[col] = 0
            out_cols.append(col)
    return type_values


def add_surrounding_counts_multi_radius(
    conflict_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    radii: Sequence[int],
    traj_frame_col: str,
    traj_id_col: str,
    traj_x_col: str,
    traj_y_col: str,
    conflict_frame_col: str,
    conflict_id1_col: str,
    conflict_id2_col: str,
    conflict_x1_col: str,
    conflict_y1_col: str,
    conflict_x2_col: str,
    conflict_y2_col: str,
    type_col: Optional[str] = None,
) -> pd.DataFrame:
    """Statistics for 'All Surrounding Objects': Vectorized distance calculation by frame across multiple radii."""
    conflict_df = conflict_df.copy()
    traj_df = traj_df.copy()

    conflict_df[conflict_id1_col] = conflict_df[conflict_id1_col].astype(str)
    conflict_df[conflict_id2_col] = conflict_df[conflict_id2_col].astype(str)
    traj_df[traj_id_col] = traj_df[traj_id_col].astype(str)

    radii = sorted(set(int(r) for r in radii))
    r2 = {r: float(r) * float(r) for r in radii}

    for r in radii:
        col = f"surrounding_count_{r}"
        if col not in conflict_df.columns:
            conflict_df[col] = 0

    type_values: List[str] = []
    if type_col:
        type_values = _init_type_columns(conflict_df, traj_df, type_col, radii)

    # Filter trajectories by relevant frames to reduce computation
    relevant_frames = conflict_df[conflict_frame_col].unique()
    traj_df = traj_df[traj_df[traj_frame_col].isin(relevant_frames)]

    grouped_conflicts = conflict_df.groupby(conflict_frame_col, sort=False)
    grouped_traj = traj_df.groupby(traj_frame_col, sort=False)

    for frame, frame_conflicts in grouped_conflicts:
        if frame not in grouped_traj.groups:
            continue
        ft = grouped_traj.get_group(frame)
        if ft.empty:
            continue

        obj_xy = ft[[traj_x_col, traj_y_col]].to_numpy(dtype=np.float32, copy=False)
        obj_id = ft[traj_id_col].to_numpy(dtype=object, copy=False).astype(str)
        obj_type = ft[type_col].astype(str).to_numpy(dtype=object, copy=False) if (type_col and type_col in ft.columns and type_values) else None

        car1_xy = frame_conflicts[[conflict_x1_col, conflict_y1_col]].to_numpy(dtype=np.float32, copy=False)
        car2_xy = frame_conflicts[[conflict_x2_col, conflict_y2_col]].to_numpy(dtype=np.float32, copy=False)
        car1_id = frame_conflicts[conflict_id1_col].to_numpy(dtype=object, copy=False).astype(str)
        car2_id = frame_conflicts[conflict_id2_col].to_numpy(dtype=object, copy=False).astype(str)

        obj_x = obj_xy[:, 0][None, :]
        obj_y = obj_xy[:, 1][None, :]
        c1x = car1_xy[:, 0:1]
        c1y = car1_xy[:, 1:2]
        c2x = car2_xy[:, 0:1]
        c2y = car2_xy[:, 1:2]

        # Key optimization: calculate squared distance once and reuse for different radii
        d2_1 = (c1x - obj_x) ** 2 + (c1y - obj_y) ** 2
        d2_2 = (c2x - obj_x) ** 2 + (c2y - obj_y) ** 2

        # Exclude the conflict pair themselves
        exclude = (obj_id[None, :] != car1_id[:, None]) & (obj_id[None, :] != car2_id[:, None])

        for r in radii:
            near = ((d2_1 <= r2[r]) | (d2_2 <= r2[r])) & exclude
            conflict_df.loc[frame_conflicts.index, f"surrounding_count_{r}"] = np.count_nonzero(near, axis=1).astype(int)

            if obj_type is not None:
                for t in type_values:
                    tmask = (obj_type == t)[None, :]
                    conflict_df.loc[frame_conflicts.index, f"surrounding_{t}_{r}"] = np.count_nonzero(near & tmask, axis=1).astype(int)

    return conflict_df


def add_related_conflict_counts_multi_radius(
    conflict_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    partners: Dict[str, Set[str]],
    radii: Sequence[int],
    traj_frame_col: str,
    traj_id_col: str,
    traj_x_col: str,
    traj_y_col: str,
    conflict_frame_col: str,
    conflict_id1_col: str,
    conflict_id2_col: str,
    conflict_x1_col: str,
    conflict_y1_col: str,
    conflict_x2_col: str,
    conflict_y2_col: str,
) -> pd.DataFrame:
    """Statistics for 'Conflict-Related Objects': Count only objects within the conflict partner set."""
    conflict_df = conflict_df.copy()
    traj_df = traj_df.copy()

    conflict_df[conflict_id1_col] = conflict_id1_col.astype(str) # Note: possible logic error in original code, should be .astype(str)
    conflict_df[conflict_id1_col] = conflict_df[conflict_id1_col].astype(str)
    conflict_df[conflict_id2_col] = conflict_df[conflict_id2_col].astype(str)
    traj_df[traj_id_col] = traj_df[traj_id_col].astype(str)

    radii = sorted(set(int(r) for r in radii))
    r2 = {r: float(r) * float(r) for r in radii}

    for r in radii:
        for col in (f"Car 1 conflictNum_{r}", f"Car 2 conflictNum_{r}", f"conflictNum_{r}"):
            if col not in conflict_df.columns:
                conflict_df[col] = 0

    relevant_frames = conflict_df[conflict_frame_col].unique()
    traj_df = traj_df[traj_df[traj_frame_col].isin(relevant_frames)]

    # Keep only objects that have ever participated in a conflict to narrow down candidate set
    all_conflict_ids = set(partners.keys())
    grouped_conflicts = conflict_df.groupby(conflict_frame_col, sort=False)
    grouped_traj = traj_df.groupby(traj_frame_col, sort=False)

    for frame, frame_conflicts in grouped_conflicts:
        if frame not in grouped_traj.groups:
            continue
        ft = grouped_traj.get_group(frame)
        if ft.empty:
            continue

        ft2 = ft[ft[traj_id_col].isin(all_conflict_ids)]
        if ft2.empty:
            continue

        obj_xy = ft2[[traj_x_col, traj_y_col]].to_numpy(dtype=np.float32, copy=False)
        obj_id = ft2[traj_id_col].to_numpy(dtype=object, copy=False).astype(str)

        car1_xy = frame_conflicts[[conflict_x1_col, conflict_y1_col]].to_numpy(dtype=np.float32, copy=False)
        car2_xy = frame_conflicts[[conflict_x2_col, conflict_y2_col]].to_numpy(dtype=np.float32, copy=False)
        car1_id = frame_conflicts[conflict_id1_col].to_numpy(dtype=object, copy=False).astype(str)
        car2_id = frame_conflicts[conflict_id2_col].to_numpy(dtype=object, copy=False).astype(str)

        obj_x = obj_xy[:, 0][None, :]
        obj_y = obj_xy[:, 1][None, :]
        c1x = car1_xy[:, 0:1]
        c1y = car1_xy[:, 1:2]
        c2x = car2_xy[:, 0:1]
        c2y = car2_xy[:, 1:2]

        d2_1 = (c1x - obj_x) ** 2 + (c1y - obj_y) ** 2
        d2_2 = (c2x - obj_x) ** 2 + (c2y - obj_y) ** 2

        for local_i, row_idx in enumerate(frame_conflicts.index):
            p1 = partners.get(car1_id[local_i], set())
            p2 = partners.get(car2_id[local_i], set())
            if not p1 and not p2:
                continue

            for r in radii:
                ids1 = obj_id[d2_1[local_i] <= r2[r]]
                ids2 = obj_id[d2_2[local_i] <= r2[r]]

                if len(ids1) <= 64:
                    c1cnt = sum((nid != car1_id[local_i]) and (nid in p1) for nid in ids1.tolist())
                else:
                    c1cnt = int(np.count_nonzero((ids1 != car1_id[local_i]) & np.isin(ids1, list(p1))))

                if len(ids2) <= 64:
                    c2cnt = sum((nid != car2_id[local_i]) and (nid in p2) for nid in ids2.tolist())
                else:
                    c2cnt = int(np.count_nonzero((ids2 != car2_id[local_i]) & np.isin(ids2, list(p2))))

                conflict_df.loc[row_idx, f"Car 1 conflictNum_{r}"] = c1cnt
                conflict_df.loc[row_idx, f"Car 2 conflictNum_{r}"] = c2cnt
                conflict_df.loc[row_idx, f"conflictNum_{r}"] = c1cnt + c2cnt

    return conflict_df


def run_one_dataset(cfg: dict) -> None:
    """Batch process by dataset."""
    name = cfg["name"]
    traj_globs: List[str] = cfg["traj_globs"]
    result_files: List[str] = cfg["result_files"]
    radii: List[int] = cfg.get("radii", [10, 20])
    traj_match: str = str(cfg.get("traj_match", "scene"))
    traj_prefer_dirs: List[str] = list(cfg.get("traj_prefer_dirs") or [])
    traj_disambiguate: str = str(cfg.get("traj_disambiguate", "priority"))
    traj_globs_by_result: dict = dict(cfg.get("traj_globs_by_result") or {})
    print_per_result = bool(cfg.get("print_per_result", True))

    traj_frame_col = cfg["traj_cols"]["frame"]
    traj_id_col = cfg["traj_cols"]["id"]
    traj_x_col = cfg["traj_cols"]["x"]
    traj_y_col = cfg["traj_cols"]["y"]
    traj_type_col = cfg["traj_cols"].get("type")

    conflict_frame_col = cfg["conflict_cols"]["frame"]
    conflict_id1_col = cfg["conflict_cols"]["id1"]
    conflict_id2_col = cfg["conflict_cols"]["id2"]
    conflict_x1_col = cfg["conflict_cols"]["x1"]
    conflict_y1_col = cfg["conflict_cols"]["y1"]
    conflict_x2_col = cfg["conflict_cols"]["x2"]
    conflict_y2_col = cfg["conflict_cols"]["y2"]

    required_result_cols = [
        "Scene",
        conflict_frame_col,
        conflict_id1_col,
        conflict_id2_col,
        conflict_x1_col,
        conflict_y1_col,
        conflict_x2_col,
        conflict_y2_col,
    ]

    base_output_dir = cfg.get("output_dir", os.path.join(os.path.dirname(__file__), "surroundings"))
    output_dir = os.path.join(base_output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    traj_cache: Dict[str, pd.DataFrame] = {}

    # Metrics for explaining discrepancies between total rows and calculated pairs:
    result_rows_total = 0       # Raw total rows in result files
    pair_key_unique = 0         # Unique pair_key count (to check for duplicates)
    scene_matched_rows = 0      # Rows where Scene found a corresponding traj file
    frame_matched_rows = 0      # Rows where Frame exists in traj (actual processed count)

    total_sur = {r: 0 for r in radii}
    total_rel = {r: 0 for r in radii}

    if traj_match == "scene_date_file":
        date_to_path = _index_date_files(traj_globs)
        traj_scene_col = str(cfg.get("traj_scene_col", "Scene"))
        traj_time_col = str(cfg.get("traj_time_col", traj_frame_col))
        conflict_time_col = str(cfg.get("conflict_time_col", conflict_frame_col))
        tol_ms = int(cfg.get("time_tolerance_ms", 20))
        chunksize = int(cfg.get("traj_read_chunksize", 200000))

        traj_usecols = [traj_scene_col, traj_time_col, traj_id_col, traj_x_col, traj_y_col]
        if traj_type_col:
            traj_usecols.append(traj_type_col)

        for result_csv in result_files:
            result_df = load_result_conflicts(result_csv, required_result_cols)
            result_rows_total += int(len(result_df))
            if "pair_key" in result_df.columns:
                pair_key_unique += int(result_df["pair_key"].nunique())

            file_tag = os.path.splitext(os.path.basename(result_csv))[0]

            df2 = result_df.copy()
            df2["__date"] = df2["Scene"].apply(_extract_date_prefix)
            date_ok = int(df2["__date"].notna().sum())
            date_bad = int(len(df2) - date_ok)
            print(f"[{name}] result={file_tag} Scene date parsing: Success={date_ok}, Fail={date_bad}")
            for k, v in df2["__date"].value_counts(dropna=True).head(8).items():
                print(f"  date={k}: Rows={int(v)}")
            for date_key, df_date in df2.groupby("__date", sort=False):
                if not date_key or (isinstance(date_key, float) and pd.isna(date_key)):
                    print(f"Warning: dataset={name} result={file_tag} Scene date missing, skipping {len(df_date)} rows")
                    continue
                traj_path = date_to_path.get(str(date_key))
                if not traj_path:
                    print(f"Warning: dataset={name} result={file_tag} date={date_key} no traj file found, skipping {len(df_date)} rows")
                    continue
                print(f"[{name}] date={date_key} Traj file: {traj_path} | Date conflict rows={len(df_date)}")

                scene_to_targets: Dict[str, np.ndarray] = {}
                for sc, g in df_date.groupby("Scene", sort=False):
                    ms = _ms_from_seconds(g[conflict_time_col]).dropna().astype(np.int64).to_numpy()
                    if ms.size:
                        scene_to_targets[str(sc)] = np.unique(np.sort(ms))
                    else:
                        scene_to_targets[str(sc)] = np.array([], dtype=np.int64)

                traj_selected = _load_songdo_traj_selected(
                    traj_path=traj_path,
                    traj_scene_col=traj_scene_col,
                    traj_time_col=traj_time_col,
                    usecols=traj_usecols,
                    scene_to_targets_sorted_ms=scene_to_targets,
                    tol_ms=tol_ms,
                    chunksize=chunksize,
                )

                for scene, conflicts_scene in df_date.groupby("Scene", sort=False):
                    traj_scene_df = traj_selected[traj_selected[traj_scene_col].astype(str) == str(scene)]
                    if traj_scene_df.empty:
                        print(
                            f"Warning: dataset={name} result={file_tag} Scene={scene} no matching time in traj, skipping {len(conflicts_scene)} rows"
                        )
                        continue

                    scene_matched_rows += int(len(conflicts_scene))

                    conflicts_scene = conflicts_scene.copy()
                    conflicts_scene["scene_matched"] = True
                    conflicts_scene["__frame_ms"] = _ms_from_seconds(conflicts_scene[conflict_time_col]).to_numpy(dtype=np.int64, na_value=-1)

                    frames_in_traj = set(pd.to_numeric(traj_scene_df["__frame_ms"], errors="coerce").dropna().astype(int).tolist())
                    frame_ok = pd.to_numeric(conflicts_scene["__frame_ms"], errors="coerce").fillna(-1).astype(int).isin(frames_in_traj)
                    frame_matched_rows += int(frame_ok.sum())

                    conflicts_scene["frame_matched"] = frame_ok.to_numpy()

                    if frame_ok.any():
                        calc_part = conflicts_scene.loc[frame_ok].copy()
                        base_cols = set(calc_part.columns)
                        calc_part = add_surrounding_counts_multi_radius(
                            conflict_df=calc_part,
                            traj_df=traj_scene_df,
                            radii=radii,
                            traj_frame_col="__frame_ms",
                            traj_id_col=traj_id_col,
                            traj_x_col=traj_x_col,
                            traj_y_col=traj_y_col,
                            conflict_frame_col="__frame_ms",
                            conflict_id1_col=conflict_id1_col,
                            conflict_id2_col=conflict_id2_col,
                            conflict_x1_col=conflict_x1_col,
                            conflict_y1_col=conflict_y1_col,
                            conflict_x2_col=conflict_x2_col,
                            conflict_y2_col=conflict_y2_col,
                            type_col=traj_type_col,
                        )

                        partners = build_partners(calc_part, conflict_id1_col, conflict_id2_col)
                        calc_part = add_related_conflict_counts_multi_radius(
                            conflict_df=calc_part,
                            traj_df=traj_scene_df,
                            partners=partners,
                            radii=radii,
                            traj_frame_col="__frame_ms",
                            traj_id_col=traj_id_col,
                            traj_x_col=traj_x_col,
                            traj_y_col=traj_y_col,
                            conflict_frame_col="__frame_ms",
                            conflict_id1_col=conflict_id1_col,
                            conflict_id2_col=conflict_id2_col,
                            conflict_x1_col=conflict_x1_col,
                            conflict_y1_col=conflict_y1_col,
                            conflict_x2_col=conflict_x2_col,
                            conflict_y2_col=conflict_y2_col,
                        )
                        computed_cols = [c for c in calc_part.columns if c not in base_cols]
                        for c in computed_cols:
                            if c not in conflicts_scene.columns:
                                conflicts_scene[c] = 0
                        if computed_cols:
                            conflicts_scene.loc[calc_part.index, computed_cols] = calc_part[computed_cols]

                        for r in radii:
                            total_sur[r] += int(calc_part[f"surrounding_count_{r}"].sum())
                            total_rel[r] += int(calc_part[f"conflictNum_{r}"].sum())

                    out_df = conflicts_scene.drop(columns=["__date", "__frame_ms"], errors="ignore")
                    out_path = os.path.join(output_dir, f"{name}_{file_tag}_{scene}_enriched.csv")
                    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n[{name}] Stats summary (Conflict source: result/*.csv)")
        print(f"Total result rows: {result_rows_total}")
        if pair_key_unique > 0:
            print(f"Total unique pair_keys: {pair_key_unique}")
        print(f"Rows matching trajectory file by Scene: {scene_matched_rows}")
        print(f"Rows matching Frame in trajectory (Actual processed): {frame_matched_rows}")
        print(f"Unmatched Scene rows: {result_rows_total - scene_matched_rows}")
        print(f"Unmatched Frame rows: {scene_matched_rows - frame_matched_rows}")
        total_pairs = frame_matched_rows
        print(f"Total conflict pairs (Mean denominator): {total_pairs}")
        for r in radii:
            avg_sur = (total_sur[r] / total_pairs) if total_pairs else 0.0
            avg_rel = (total_rel[r] / total_pairs) if total_pairs else 0.0
            print(f"R={r}m: Total surrounding objects={total_sur[r]}, Mean per pair={avg_sur:.2f} | Total related objects={total_rel[r]}, Mean per pair={avg_rel:.2f}")
        return

    global_scene_to_paths = index_traj_files(traj_globs)
    scene_to_paths_cache: Dict[str, Dict[str, List[str]]] = {}

    for result_csv in result_files:
        result_df = load_result_conflicts(result_csv, required_result_cols)
        file_tag = os.path.splitext(os.path.basename(result_csv))[0]
        file_rows_total = int(len(result_df))
        file_pair_key_unique = int(result_df["pair_key"].nunique()) if "pair_key" in result_df.columns else 0
        file_scene_matched_rows = 0
        file_frame_matched_rows = 0
        file_total_sur = {r: 0 for r in radii}
        file_total_rel = {r: 0 for r in radii}

        result_rows_total += file_rows_total
        pair_key_unique += file_pair_key_unique

        scene_to_paths = global_scene_to_paths
        if traj_globs_by_result and file_tag in traj_globs_by_result:
            if file_tag not in scene_to_paths_cache:
                local_globs = traj_globs_by_result[file_tag]
                if isinstance(local_globs, str):
                    local_globs = [local_globs]
                scene_to_paths_cache[file_tag] = index_traj_files(list(local_globs))
            scene_to_paths = scene_to_paths_cache[file_tag]

        traj_df_for_result: Optional[pd.DataFrame] = None
        if traj_match == "result_file":
            traj_path = _select_traj_path_by_result_tag(file_tag, scene_to_paths)
            if traj_path is None:
                print(f"Warning: dataset={name} result={file_tag} no traj file found, skipping {len(result_df)} rows")
                if print_per_result:
                    print(f"\n[{name}] Item stats result={file_tag}")
                    print(f"result rows: {file_rows_total}")
                    if file_pair_key_unique > 0:
                        print(f"Unique pair_keys: {file_pair_key_unique}")
                    print(f"Rows matching traj by Scene: {file_scene_matched_rows}")
                    print(f"Rows matching Frame in traj: {file_frame_matched_rows}")
                    print(f"Unmatched Scene: {file_rows_total - file_scene_matched_rows}")
                    print(f"Unmatched Frame: {file_scene_matched_rows - file_frame_matched_rows}")
                    total_pairs = file_frame_matched_rows
                    print(f"Total conflict pairs: {total_pairs}")
                    for r in radii:
                        avg_sur = (file_total_sur[r] / total_pairs) if total_pairs else 0.0
                        avg_rel = (file_total_rel[r] / total_pairs) if total_pairs else 0.0
                        print(f"R={r}m: Total surrounding={file_total_sur[r]}, Mean={avg_sur:.2f} | Total related={file_total_rel[r]}, Mean={avg_rel:.2f}")
                continue
            if file_tag not in traj_cache:
                traj_cache[file_tag] = pd.read_csv(traj_path)
            traj_df_for_result = traj_cache[file_tag]

        for scene, conflicts_scene in result_df.groupby("Scene", sort=False):
            if traj_match == "scene":
                cand_paths = _candidate_paths_for_scene(scene, scene_to_paths)
                if not cand_paths:
                    print(f"Warning: dataset={name} result={file_tag} Scene={scene} no traj file, skipping {len(conflicts_scene)} rows")
                    continue
                ranked_paths = _rank_paths_by_prefer_dirs(cand_paths, traj_prefer_dirs)
                traj_path = ranked_paths[0]
                if len(ranked_paths) > 1:
                    if traj_disambiguate == "frame":
                        target_frames = set(
                            pd.to_numeric(conflicts_scene[conflict_frame_col], errors="coerce").dropna().astype(int).unique().tolist()
                        )
                        best_path = ranked_paths[0]
                        best_cov = -1
                        for p in ranked_paths:
                            cov = _frame_coverage_count(p, traj_frame_col, target_frames)
                            if cov > best_cov:
                                best_cov = cov
                                best_path = p
                                if best_cov == len(target_frames):
                                    break
                        traj_path = best_path
                        print(f"Warning: Scene={scene} matched {len(ranked_paths)} files, selecting by frame coverage: {traj_path} | Coverage={best_cov}/{len(target_frames)}")
                    else:
                        print(f"Warning: Scene={scene} matched {len(ranked_paths)} files, selecting by priority: {traj_path}")

                if traj_path not in traj_cache:
                    traj_cache[traj_path] = pd.read_csv(traj_path)
                traj_df = traj_cache[traj_path]
            elif traj_match == "result_scene_prefix_case":
                scene_prefix, scene_case = _split_scene_prefix_case(scene, sep=str(cfg.get("scene_sep", "_")))
                if not scene_prefix or not scene_case:
                    print(f"Warning: dataset={name} result={file_tag} Scene={scene} split/case parsing failed, skipping {len(conflicts_scene)} rows")
                    continue

                traj_tag = f"{file_tag}_{scene_prefix}"
                traj_path = _select_traj_path_by_result_tag(traj_tag, scene_to_paths)
                if traj_path is None:
                    print(f"Warning: dataset={name} result={file_tag} Scene={scene} traj_tag={traj_tag} no traj file, skipping {len(conflicts_scene)} rows")
                    continue

                if traj_path not in traj_cache:
                    traj_cache[traj_path] = pd.read_csv(traj_path)
                traj_df = traj_cache[traj_path]

                case_col = str(cfg.get("traj_case_col", "case_id"))
                if case_col not in traj_df.columns:
                    raise ValueError(f"dataset={name} traj_match=result_scene_prefix_case requires column {case_col} | file={traj_path}")

                case_key = str(scene_case).strip()
                m = re.match(r"^(\d+)", case_key)
                if m:
                    case_key = m.group(1)
                case_int: Optional[int] = None
                try:
                    case_int = int(case_key)
                except Exception:
                    case_int = None

                if case_int is not None:
                    ser_num = pd.to_numeric(traj_df[case_col], errors="coerce")
                    mask = ser_num.eq(case_int)
                    if bool(mask.any()):
                        traj_df = traj_df[mask]
                    else:
                        ser_str = traj_df[case_col].astype(str).str.strip()
                        ser_str = ser_str.str.replace(r"\.0+$", "", regex=True)
                        traj_df = traj_df[ser_str == str(case_int)]
                else:
                    traj_df = traj_df[traj_df[case_col].astype(str).str.strip() == case_key]
                if traj_df.empty:
                    print(f"Warning: dataset={name} result={file_tag} Scene={scene} case_id={case_key} no match in traj, skipping {len(conflicts_scene)} rows")
                    continue
            else:
                traj_df = traj_df_for_result
                if traj_df is None:
                    continue
                scene_col = cfg.get("traj_scene_col")
                if not scene_col and "Scene" in traj_df.columns:
                    scene_col = "Scene"
                if scene_col and scene_col in traj_df.columns:
                    s = str(scene)
                    ser = traj_df[scene_col].astype(str)
                    if (ser == s).any():
                        traj_df = traj_df[ser == s]

            scene_matched_rows += int(len(conflicts_scene))
            file_scene_matched_rows += int(len(conflicts_scene))

            frames_in_traj = set(pd.to_numeric(traj_df[traj_frame_col], errors="coerce").dropna().astype(int).tolist())
            frame_ok = pd.to_numeric(conflicts_scene[conflict_frame_col], errors="coerce").fillna(-1).astype(int).isin(frames_in_traj)
            frame_matched_rows += int(frame_ok.sum())
            file_frame_matched_rows += int(frame_ok.sum())

            conflicts_scene = conflicts_scene.copy()
            conflicts_scene["scene_matched"] = True
            conflicts_scene["frame_matched"] = frame_ok.to_numpy()

            if frame_ok.any():
                calc_part = conflicts_scene.loc[frame_ok].copy()
                base_cols = set(calc_part.columns)
                calc_part = add_surrounding_counts_multi_radius(
                    conflict_df=calc_part,
                    traj_df=traj_df,
                    radii=radii,
                    traj_frame_col=traj_frame_col,
                    traj_id_col=traj_id_col,
                    traj_x_col=traj_x_col,
                    traj_y_col=traj_y_col,
                    conflict_frame_col=conflict_frame_col,
                    conflict_id1_col=conflict_id1_col,
                    conflict_id2_col=conflict_id2_col,
                    conflict_x1_col=conflict_x1_col,
                    conflict_y1_col=conflict_y1_col,
                    conflict_x2_col=conflict_x2_col,
                    conflict_y2_col=conflict_y2_col,
                    type_col=traj_type_col,
                )

                partners = build_partners(calc_part, conflict_id1_col, conflict_id2_col)
                calc_part = add_related_conflict_counts_multi_radius(
                    conflict_df=calc_part,
                    traj_df=traj_df,
                    partners=partners,
                    radii=radii,
                    traj_frame_col=traj_frame_col,
                    traj_id_col=traj_id_col,
                    traj_x_col=traj_x_col,
                    traj_y_col=traj_y_col,
                    conflict_frame_col=conflict_frame_col,
                    conflict_id1_col=conflict_id1_col,
                    conflict_id2_col=conflict_id2_col,
                    conflict_x1_col=conflict_x1_col,
                    conflict_y1_col=conflict_y1_col,
                    conflict_x2_col=conflict_x2_col,
                    conflict_y2_col=conflict_y2_col,
                )
                computed_cols = [c for c in calc_part.columns if c not in base_cols]
                for c in computed_cols:
                    if c not in conflicts_scene.columns:
                        conflicts_scene[c] = 0
                if computed_cols:
                    conflicts_scene.loc[calc_part.index, computed_cols] = calc_part[computed_cols]

                for r in radii:
                    total_sur[r] += int(calc_part[f"surrounding_count_{r}"].sum())
                    total_rel[r] += int(calc_part[f"conflictNum_{r}"].sum())
                    file_total_sur[r] += int(calc_part[f"surrounding_count_{r}"].sum())
                    file_total_rel[r] += int(calc_part[f"conflictNum_{r}"].sum())

            out_path = os.path.join(output_dir, f"{name}_{file_tag}_{scene}_enriched.csv")
            conflicts_scene.to_csv(out_path, index=False, encoding="utf-8-sig")

        if print_per_result:
            print(f"\n[{name}] Item stats result={file_tag}")
            print(f"result rows: {file_rows_total}")
            if file_pair_key_unique > 0:
                print(f"Unique pair_keys: {file_pair_key_unique}")
            print(f"Rows matching traj by Scene: {file_scene_matched_rows}")
            print(f"Rows matching Frame in traj: {file_frame_matched_rows}")
            print(f"Unmatched Scene rows: {file_rows_total - file_scene_matched_rows}")
            print(f"Unmatched Frame rows: {file_scene_matched_rows - file_frame_matched_rows}")
            total_pairs = file_frame_matched_rows
            print(f"Total conflict pairs (Mean denominator): {total_pairs}")
            for r in radii:
                avg_sur = (file_total_sur[r] / total_pairs) if total_pairs else 0.0
                avg_rel = (file_total_rel[r] / total_pairs) if total_pairs else 0.0
                print(f"R={r}m: Total surrounding={file_total_sur[r]}, Mean={avg_sur:.2f} | Total related={file_total_rel[r]}, Mean={avg_rel:.2f}")

    print(f"\n[{name}] Overall Statistics Summary")
    print(f"Aggregate result rows: {result_rows_total}")
    if pair_key_unique > 0:
        print(f"Aggregate unique pair_keys: {pair_key_unique}")
    print(f"Rows matching traj by Scene: {scene_matched_rows}")
    print(f"Rows matching Frame in traj (Processed): {frame_matched_rows}")
    print(f"Total unmatched Scene rows: {result_rows_total - scene_matched_rows}")
    print(f"Total unmatched Frame rows: {scene_matched_rows - frame_matched_rows}")

    total_pairs = frame_matched_rows
    print(f"Final conflict pair count (Mean denominator): {total_pairs}")
    for r in radii:
        avg_sur = (total_sur[r] / total_pairs) if total_pairs else 0.0
        avg_rel = (total_rel[r] / total_pairs) if total_pairs else 0.0
        print(f"R={r}m: Total surrounding={total_sur[r]}, Mean={avg_sur:.2f} | Total related={total_rel[r]}, Mean={avg_rel:.2f}")


if __name__ == "__main__":
    """
    Dataset Configuration Block
    - Define dataset-specific column names and paths here.
    """

    DATASETS = [
        # Example: INTERACTION (predicted)
        {
            "name": "INTERACTION_predicted",
            "traj_match": "result_scene_prefix_case",
            "traj_globs": [r"G:\Dataset\INTERACTION\Trajectory\*.csv"],
            "scene_sep": "_",
            "traj_case_col": "case_id",
            "result_files": [
                r"F:\GIS_project\Intersection_Conflict_Calculation\result\DR_USA_Intersection_EP0.csv",
                r"F:\GIS_project\Intersection_Conflict_Calculation\result\DR_USA_Intersection_EP1.csv",
                r"F:\GIS_project\Intersection_Conflict_Calculation\result\DR_USA_Intersection_GL.csv",
                r"F:\GIS_project\Intersection_Conflict_Calculation\result\DR_USA_Intersection_MA.csv",
            ],
            "radii": [10, 20],
            "traj_cols": {"frame": "frame_id", "id": "track_id", "x": "x", "y": "y", "type": "agent_type"},
            "conflict_cols": {
                "frame": "Frame",
                "id1": "Car 1 ID",
                "id2": "Car 2 ID",
                "x1": "Car 1 X",
                "y1": "Car 1 Y",
                "x2": "Car 2 X",
                "y2": "Car 2 Y",
            },
        },
        
        # Other Dataset
                # # CitySim
        # {
        #     "name": "CitySim",
        #     "traj_globs": [r"G:\Dataset\CitySim\Trajectories\*.csv"],
        #     "result_files": [
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\IntersectionA.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\IntersectionB.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\IntersectionD.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\IntersectionE.csv",
        #     ],
        #     "radii": [10, 20],
        #     "traj_cols": {"frame": "frameNum", "id": "carId", "x": "xUtm", "y": "yUtm"},
        #     "conflict_cols": {
        #         "frame": "Frame",
        #         "id1": "Car 1 ID",
        #         "id2": "Car 2 ID",
        #         "x1": "Car 1 X",
        #         "y1": "Car 1 Y",
        #         "x2": "Car 2 X",
        #         "y2": "Car 2 Y",
        #     },
        # },
        #
        # # SIND
        # {
        #     "name": "SinD",
        #     "traj_globs": [
        #         r"G:\Dataset\SinD\mergedTraj\Changchun\**\*.csv",
        #         r"G:\Dataset\SinD\mergedTraj\Chongqing\**\*.csv",
        #         r"G:\Dataset\SinD\mergedTraj\Tianjin\**\*.csv",
        #         r"G:\Dataset\SinD\mergedTraj\Xi'an\**\*.csv",
        #     ],
        #     "result_files": [
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\Changchun.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\Chongqing.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\Tianjin.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\Xi'an.csv",
        #     ],
        #     "radii": [10, 20],
        #     "traj_cols": {"frame": "frame_id", "id": "track_id", "x": "x", "y": "y", "type": "agent_type"},
        #     "conflict_cols": {
        #         "frame": "Frame",
        #         "id1": "Car 1 ID",
        #         "id2": "Car 2 ID",
        #         "x1": "Car 1 X",
        #         "y1": "Car 1 Y",
        #         "x2": "Car 2 X",
        #         "y2": "Car 2 Y",
        #     },
        # },
        #
        # # FLUID
        # {
        #     "name": "FLUID",
        #     "traj_globs": [
        #         r"F:\GIS_project\交叉口冲突计算\syj\result\*.csv",
        #         r"F:\GIS_project\交叉口冲突计算\zszy\result\*.csv",
        #         r"F:\GIS_project\交叉口冲突计算\gsb\result\*.csv",
        #     ],
        #     "traj_globs_by_result": {
        #         "FIDRT": [r"F:\GIS_project\交叉口冲突计算\syj\result\*.csv"],
        #         "FI": [r"F:\GIS_project\交叉口冲突计算\zszy\result\*.csv"],
        #         "TI": [r"F:\GIS_project\交叉口冲突计算\gsb\result\*.csv"],
        #     },
        #     "result_files": [
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\FIDRT.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\FI.csv",
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\TI.csv",
        #     ],
        #     "radii": [10, 20],
        #     "traj_cols": {"frame": "frame", "id": "id", "x": "cx_m", "y": "cy_m", "type": "type"},
        #     "conflict_cols": {
        #         "frame": "Frame",
        #         "id1": "Car 1 ID",
        #         "id2": "Car 2 ID",
        #         "x1": "Car 1 X",
        #         "y1": "Car 1 Y",
        #         "x2": "Car 2 X",
        #         "y2": "Car 2 Y",
        #     },
        # },

        # # Songdo Traffic
        # {
        #     "name": "Songdo Traffic",
        #     "traj_match": "scene_date_file",
        #     "traj_scene_col": "Scene",
        #     "traj_time_col": "time",
        #     "conflict_time_col": "Time",
        #     "time_tolerance_ms": 20,
        #     "traj_read_chunksize": 200000,
        #     "traj_globs": [
        #         r"G:\Dataset\SongdoTraffic\data\2022-10-04.csv",
        #         r"G:\Dataset\SongdoTraffic\data\2022-10-05.csv",
        #         r"G:\Dataset\SongdoTraffic\data\2022-10-06.csv",
        #         r"G:\Dataset\SongdoTraffic\data\2022-10-07.csv",
        #     ],
        #     "result_files": [
        #         r"F:\GIS_project\Intersection_Conflict_Calculation\result\SongdoTraffic.csv",
        #     ],
        #     "radii": [10, 20],
        #     "traj_cols": {"frame": "time", "id": "Vehicle_ID", "x": "Local_X", "y": "Local_Y", "type": "Vehicle_Class"},
        #     "conflict_cols": {
        #         "frame": "Time",
        #         "id1": "Car 1 ID",
        #         "id2": "Car 2 ID",
        #         "x1": "Car 1 X",
        #         "y1": "Car 1 Y",
        #         "x2": "Car 2 X",
        #         "y2": "Car 2 Y",
        #     },
        # },
    ]

    for ds in DATASETS:
        run_one_dataset(ds)