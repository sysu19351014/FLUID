# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/06 14:15
@Author  : Terry_CYY
@File    : 3_dropErrorfromSSM.py
@IDE     : PyCharm
@Function: Filter abnormal trajectories based on Surrogate Safety Measures (SSM), 
           including TTC and PET (DGT). It identifies outliers from calculated SSM results 
           and removes corresponding frames/IDs from the original dataset.
"""

import numpy as np
import pandas as pd
import os
import sys
import subprocess
from tqdm import tqdm
from collections import defaultdict


def drop_fault_object(data):
    """
    Remove faulty objects based on duration, confidence, and trajectory length.
    Now optimized for all object types due to improved tracking precision.
    """
    # 1. Remove IDs with a total duration (max time - min time) less than 1 second
    # Vulnerable Road Users (VRUs) are handled specifically regarding confidence
    valid_types = ['pedestrian', 'moped']
    
    # Aggregate statistics for all IDs in one step
    id_stats = data.groupby('id').agg(
        min_time=('time', 'min'),
        max_time=('time', 'max'),
        has_valid_type=('type', lambda x: x.isin(valid_types).any())
    ).reset_index()

    # Calculate duration
    id_stats['duration'] = id_stats['max_time'] - id_stats['min_time']

    # Identify invalid IDs (Duration <= 1s)
    invalid_ids_duration = id_stats[id_stats['duration'] <= 1]['id']
    data = data[~data['id'].isin(invalid_ids_duration)]

    # 2. Remove non-VRU IDs with an average confidence below 0.5
    invalid_ids_conf = (
        data.groupby('id')
        .filter(lambda g:
                ~g['type'].isin(valid_types).any() and
                g['confidence'][~g['confidence'].isna()].mean() < 0.5
                )['id']
        .unique()
    )
    data = data[~data['id'].isin(invalid_ids_conf)]

    # 3. Remove trajectories with extremely short spatial lengths (VRU specific)
    def calc_traj_length(group):
        """Calculate the cumulative spatial length of a trajectory."""
        group = group.sort_values('frame')
        coords = group[['cx_m', 'cy_m']].values
        if len(coords) < 2:
            return 0
        # Calculate Euclidean distance between adjacent points
        deltas = np.diff(coords, axis=0)
        dists = np.linalg.norm(deltas, axis=1)
        return dists.sum()

    type_vru = ['pedestrian', 'moped']
    df_vru = data[data['type'].isin(type_vru)]
    
    if not df_vru.empty:
        traj_lengths = df_vru.groupby('id').apply(calc_traj_length)
        # Identify VRU IDs with trajectory length less than 3 meters
        short_vru_ids = traj_lengths[traj_lengths < 3].index
        df_vru_filtered = df_vru[~df_vru['id'].isin(short_vru_ids)]
        
        # Recombine with non-VRU data
        df_other = data[~data['type'].isin(type_vru)]
        data = pd.concat([df_vru_filtered, df_other], ignore_index=True)

    return data


# ============ Abnormal Trajectory Filtering via TTC (Time-to-Collision) ==========

def filter_abnormal_trajectories(traj_dir, ssm_dir, output_dir, abnormal_threshold=0.9):
    """
    Main function: Removes abnormal trajectories that consistently maintain a TTC of -1 
    (indicating persistent bounding box overlap/collision state) throughout their lifetime.
    
    abnormal_threshold: Threshold ratio of abnormal frames to trigger removal.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_traj_dir = output_dir  # Save cleaned trajectories directly to output
    os.makedirs(clean_traj_dir, exist_ok=True)

    traj_files = [f for f in os.listdir(traj_dir) if f.endswith('.csv')]

    # Initial Pass: General Cleanup
    for traj_file in tqdm(traj_files, desc='Initial Trajectory Cleanup'):
        traj_path = os.path.join(traj_dir, traj_file)
        traj_df = pd.read_csv(traj_path)
        traj_df.sort_values(by=['id', 'frame'], inplace=True)
        
        # Apply basic fault object filtering
        traj_df = drop_fault_object(traj_df)
        
        # Define standardized column order
        standard_cols = [
            "frame", "id", "cx", "cy", "w", "h", "r", "confidence", "type", "w_m", "h_m",
            "time", "speed", "cx_m", "cy_m", "smooth_cx", "smooth_cy", "smooth_r", "r_align",
            "isReal", "length_med", "width_med", "w_med", "h_med", "course", "yaw", "speed_smooth", "vx", "vy", "ax", "ay",
            "entry_direction", "exit_direction", "overall_direction"
        ]
        
        cols_to_use = [c for c in standard_cols if c in traj_df.columns]
        cols_to_use += [c for c in traj_df.columns if c not in standard_cols]
        
        traj_df = traj_df[cols_to_use]
        traj_df.to_csv(os.path.join(clean_traj_dir, traj_file), index=False)

    # Step 1: Run 2D-SSM Calculation (External Script)
    print("Calculating 2D-SSM for TTC analysis...")
    cmd_ttc = [
        sys.executable,
        'UAVIntersectionTwoDimSSM.py',
        "--input_dir", clean_traj_dir,
        "--output_dir", ssm_dir,
    ]
    subprocess.run(cmd_ttc, check=True)

    # Step 2: Refine Cleanup based on TTC results
    traj_files = [f for f in os.listdir(clean_traj_dir) if f.endswith('.csv')]
    for traj_file in tqdm(traj_files, desc='Filtering via TTC Analysis'):
        traj_path = os.path.join(clean_traj_dir, traj_file)
        ssm_path = os.path.join(ssm_dir, traj_file.replace('.csv', '_2D-SSM.csv'))
        
        traj_df = pd.read_csv(traj_path)
        if not os.path.exists(ssm_path):
            continue
        ttc_df = pd.read_csv(ssm_path)

        # Separate VRUs and motor vehicles (TTC logic applied mainly to motor vehicles)
        vru_df = traj_df[traj_df['type'].isin(['moped', 'pedestrian'])]
        motor_df = traj_df[~traj_df['type'].isin(['moped', 'pedestrian'])]

        # Identify candidate IDs where TTC is frequently -1
        candidate_ids = identify_candidate_abnormal_ids(ttc_df)
        abnormal_stats = calculate_abnormal_ratio(ttc_df, candidate_ids)
        traj_spans = calculate_traj_spans(motor_df, candidate_ids)

        # Identify strictly abnormal IDs
        abnormal_ids = set()
        for car_id in candidate_ids:
            if car_id not in abnormal_stats:
                continue
            _, total_frames, ratio = abnormal_stats[car_id]
            # Flag as abnormal if ratio exceeds threshold and has at least 10 frames
            if ratio >= abnormal_threshold and total_frames >= 10:
                abnormal_ids.add(car_id)

        if not abnormal_ids:
            motor_df = pd.concat([motor_df, vru_df], ignore_index=True)
            motor_df.sort_values(by=['id', 'frame'], inplace=True)
            motor_df.to_csv(os.path.join(clean_traj_dir, traj_file), index=False)
            continue

        # Detect associated conflict pairs
        conflict_pairs = find_conflict_pairs(ttc_df, abnormal_ids)

        # Determine final IDs to remove (resolve conflicts by keeping the longer trajectory)
        to_remove = set()
        for pair in conflict_pairs:
            id1, id2 = pair
            span1, span2 = traj_spans.get(id1, 0), traj_spans.get(id2, 0)
            if span1 >= span2:
                to_remove.add(id2)
            else:
                to_remove.add(id1)

        # Handle isolated abnormal IDs
        isolated_abnormal = abnormal_ids.copy()
        for pair in conflict_pairs:
            isolated_abnormal.discard(pair[0])
            isolated_abnormal.discard(pair[1])
        to_remove.update(isolated_abnormal)

        # Remove IDs from both trajectory and TTC results
        if to_remove:
            print(f"Removing abnormal IDs from {traj_file}: {to_remove}")
            motor_df = motor_df[~motor_df['id'].isin(to_remove)]
            clean_ttc_df = ttc_df[~(ttc_df['Car 1 ID'].isin(to_remove) | ttc_df['Car 2 ID'].isin(to_remove))]
            clean_ttc_df.to_csv(ssm_path, index=False)

        # Recombine and save
        final_df = pd.concat([motor_df, vru_df], ignore_index=True)
        final_df.drop_duplicates(subset=['id', 'frame'], inplace=True)
        final_df.sort_values(by=['id', 'frame'], inplace=True)
        final_df.to_csv(os.path.join(clean_traj_dir, traj_file), index=False)

    # Step 3: Recalculate DGT/PET after TTC-based cleanup
    print("Recalculating PET for finalized trajectories...")
    cmd_dgt = [
        sys.executable,
        'UAVPETBoundary.py',
        "--input_dir", clean_traj_dir,
        "--output_dir", ssm_dir,
    ]
    try:
        subprocess.run(cmd_dgt, check=True)
    except subprocess.CalledProcessError as e:
        print(f"PET Calculation Error (Code {e.returncode}):\n{e.stderr}")

    # Step 4: Final Refinement based on PET results (Ghost Identification)
    traj_files = [f for f in os.listdir(clean_traj_dir) if f.endswith('.csv')]
    for traj_file in tqdm(traj_files, desc='Final PET-based Refinement'):
        traj_path = os.path.join(clean_traj_dir, traj_file)
        pet_path = os.path.join(ssm_dir, traj_file.replace('.csv', '_PET.csv'))
        
        if not os.path.exists(pet_path):
            continue
            
        traj_df = pd.read_csv(traj_path)
        pet_df = pd.read_csv(pet_path)
        
        pet_clean, traj_clean, removed_ids, _ = clean_by_pet_zero(
            pet_df, traj_df, threshold=0.1, abs_diff=True, use_all_rows=False
        )
        
        if removed_ids:
            print(f"Removed ghost IDs from {traj_file} via PET check: {removed_ids}")
            pet_clean.to_csv(pet_path, index=False)
            traj_clean.to_csv(traj_path, index=False)


# -------------- Helper Functions for TTC Abnormal Filtering ----------

def identify_candidate_abnormal_ids(ttc_df):
    """Identify IDs involved in any record where TTC = -1."""
    neg_ttc_df = ttc_df[ttc_df['TTC'] == -1]
    if neg_ttc_df.empty:
        return set()
    return set(neg_ttc_df['Car 1 ID'].unique()) | set(neg_ttc_df['Car 2 ID'].unique())


def calculate_abnormal_ratio(ttc_df, candidate_ids):
    """Calculate ratio of frames where an object is in a TTC = -1 state."""
    car1_df = ttc_df[ttc_df['Car 1 ID'].isin(candidate_ids)]
    car2_df = ttc_df[ttc_df['Car 2 ID'].isin(candidate_ids)]
    
    combined = pd.concat([
        car1_df[['Frame', 'Car 1 ID', 'TTC']].rename(columns={'Car 1 ID': 'id'}),
        car2_df[['Frame', 'Car 2 ID', 'TTC']].rename(columns={'Car 2 ID': 'id'})
    ])
    
    stats = {}
    for car_id, group in combined.groupby('id'):
        total = group['Frame'].nunique()
        abnormal = group[group['TTC'] == -1]['Frame'].nunique()
        stats[car_id] = (abnormal, total, abnormal / total if total > 0 else 0)
    return stats


def calculate_traj_spans(traj_df, candidate_ids):
    """Calculate the lifetime (frame count) of candidate IDs."""
    candidate_df = traj_df[traj_df['id'].isin(candidate_ids)]
    if candidate_df.empty:
        return {}
    return candidate_df.groupby('id')['frame'].agg(lambda x: x.max() - x.min()).to_dict()


def find_conflict_pairs(ttc_df, abnormal_ids):
    """Identify pairs of abnormal vehicles in persistent conflict."""
    conflict_df = ttc_df[
        (ttc_df['TTC'] == -1) &
        (ttc_df['Car 1 ID'].isin(abnormal_ids)) &
        (ttc_df['Car 2 ID'].isin(abnormal_ids))
    ]
    pairs = set()
    for _, row in conflict_df.iterrows():
        id1, id2 = row['Car 1 ID'], row['Car 2 ID']
        pairs.add(tuple(sorted((id1, id2))))
    return list(pairs)


# -------------- Helper Functions for PET/DGT Refinement ----------

def span_seconds(s: pd.Series) -> float:
    """Return the numerical span (max - min) of a series."""
    s = pd.to_numeric(s, errors='coerce')
    return float(s.max() - s.min()) if s.notna().any() else np.nan


def clean_by_pet_zero(df_pet: pd.DataFrame,
                      df_traj: pd.DataFrame,
                      threshold: float = 0.1,
                      abs_diff: bool = True,
                      use_all_rows: bool = False):
    """
    Identifies 'Ghost' trajectories where the total trajectory duration 
    is almost identical to the time spent in an intersection overlap (PET=0).
    """
    base = df_pet if use_all_rows else df_pet[df_pet['PET'] == 0]
    if base.empty:
        return df_pet.copy(), df_traj.copy(), set(), pd.DataFrame()

    # Calculate time span delta in PET results
    pair_delta = base.groupby(['id1', 'id2'], as_index=False).agg(
        delta1=('time1', span_seconds),
        delta2=('time2', span_seconds)
    )
    
    # Calculate actual duration in trajectory data
    dt_by_id = df_traj.groupby('id', as_index=False).agg(dt=('time', span_seconds))
    
    # Merge and compare
    pair_delta = (
        pair_delta
        .merge(dt_by_id.rename(columns={'id': 'id1', 'dt': 'dt1'}), on='id1', how='left')
        .merge(dt_by_id.rename(columns={'id': 'id2', 'dt': 'dt2'}), on='id2', how='left')
    )
    
    if abs_diff:
        pair_delta['flag1'] = (pair_delta['dt1'] - pair_delta['delta1']).abs() < threshold
        pair_delta['flag2'] = (pair_delta['dt2'] - pair_delta['delta2']).abs() < threshold
    else:
        pair_delta['flag1'] = (pair_delta['dt1'] - pair_delta['delta1']) < threshold
        pair_delta['flag2'] = (pair_delta['dt2'] - pair_delta['delta2']) < threshold

    removed_ids = set(pair_delta.loc[pair_delta['flag1'], 'id1']).union(
                  set(pair_delta.loc[pair_delta['flag2'], 'id2']))
                  
    df_traj_clean = df_traj[~df_traj['id'].isin(removed_ids)].copy()
    df_pet_clean = df_pet[~(df_pet['id1'].isin(removed_ids) | df_pet['id2'].isin(removed_ids))].copy()
    
    evidence = pair_delta[['id1', 'id2', 'delta1', 'dt1', 'flag1', 'delta2', 'dt2', 'flag2']].copy()
    return df_pet_clean, df_traj_clean, removed_ids, evidence


if __name__ == '__main__':
    # Configuration for batch processing
    scene_folders = ['.']
    
    for folder in tqdm(scene_folders, desc='Processing Scenes'):
        TRAJ_DIR = os.path.join(folder, 'result')
        SSM_DIR = os.path.join(folder, 'conflict')
        OUTPUT_DIR = os.path.join(folder, 'final')

        filter_abnormal_trajectories(TRAJ_DIR, SSM_DIR, OUTPUT_DIR)