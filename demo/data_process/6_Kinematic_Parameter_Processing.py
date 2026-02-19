# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/04 17:59
@Author  : Terry_CYY
@File    : 6_Kinematic_Parameter_Processing.py
@IDE     : PyCharm
@Function: Perform Kalman filtering and RTS smoothing for velocity and acceleration.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def smooth_speed(df_one_id: pd.DataFrame,
                 fps: float = 10.0,
                 meas_std: float = 0.5,
                 accel_std: float = 3.5) -> pd.DataFrame:
    """
    Apply Kalman Filter and Rauch-Tung-Striebel (RTS) smoother to velocity data for a single object.
    """
    dt = 1.0 / fps
    # State vector dim_x=2: [position/velocity_state, acceleration_state] 
    # Here we treat speed as the primary state to be filtered
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # State transition matrix
    kf.F = np.array([[1., dt],
                     [0., 1.]])
    # Measurement function
    kf.H = np.array([[1., 0.]])
    # Measurement noise covariance
    kf.R = np.array([[meas_std ** 2]])
    # Process noise covariance
    kf.Q = Q_discrete_white_noise(2, dt, accel_std ** 2)
    # Initial state
    kf.x = np.array([df_one_id['speed'].iloc[0], 0.0])
    # Initial covariance matrix
    kf.P *= 25.0

    zs = df_one_id['speed'].values
    # Forward pass (Filtering)
    mu, cov, *_ = kf.batch_filter(zs)
    # Backward pass (Smoothing)
    xs, _, _, _ = kf.rts_smoother(mu, cov)

    return pd.DataFrame({'speed_smooth': xs[:, 0]}, index=df_one_id.index)


def apply_smooth(df: pd.DataFrame, id_col: str = 'id', fps: float = 10.0) -> pd.DataFrame:
    """
    Group the dataframe by ID and apply speed smoothing.
    """
    if 'speed_smooth' in df.columns:
        return df

    # Group by object ID and apply Kalman smoothing
    smoothed = (df.groupby(id_col, group_keys=False)
                .apply(lambda g: smooth_speed(g, fps=fps)))
    smoothed['speed_smooth'] = smoothed['speed_smooth'].round(2)
    return df.join(smoothed)


def calculate_acceleration(df: pd.DataFrame, id_col: str = 'id', frame_col: str = 'frame', 
                           time_col: str = 'time', speed_col: str = 'speed', 
                           accel_col: str = 'accel', fps: float = 10.0) -> pd.DataFrame:
    """
    Calculate acceleration based on the change in velocity over time.
    """
    if time_col not in df.columns:
        df = df.copy()
        df['_temp_time'] = df[frame_col] / fps
        time_col = '_temp_time'

    # Ensure data is sorted by ID and time
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    def calc_group_accel(group: pd.DataFrame) -> pd.DataFrame:
        speeds = group[speed_col].values
        times = group[time_col].values

        if len(group) <= 1:
            group[accel_col] = 0.0
            return group

        # Calculate time difference (dt) using central difference
        dt = np.zeros_like(times)
        dt[1:-1] = (times[2:] - times[:-2]) / 2.0
        dt[0] = times[1] - times[0]
        dt[-1] = times[-1] - times[-2]
        dt[dt == 0] = 1e-6 # Avoid division by zero

        # Calculate velocity difference (dv) using central difference
        dv = np.zeros_like(speeds)
        dv[1:-1] = (speeds[2:] - speeds[:-2]) / 2.0
        dv[0] = speeds[1] - speeds[0]
        dv[-1] = speeds[-1] - speeds[-2]

        group[accel_col] = np.round(dv / dt, 2)
        return group

    df = df.groupby(id_col, group_keys=False).apply(calc_group_accel)

    if '_temp_time' in df.columns:
        df = df.drop(columns=['_temp_time'])

    return df


def add_kinematic_params(df: pd.DataFrame, fps: float = 10.0, smooth: bool = True) -> pd.DataFrame:
    """
    Calculate and append kinematic parameters (vx, vy, ax, ay) to the dataframe.
    """
    df = df.sort_values(by=['id', 'frame']).reset_index(drop=True)

    if smooth:
        df = apply_smooth(df, id_col='id', fps=fps)

    # Determine which yaw/heading column to use
    yaw_src = 'yaw'
    if yaw_src not in df.columns and 'final_motion_yaw' in df.columns:
        yaw_src = 'final_motion_yaw'

    # Calculate smoothed acceleration
    df = calculate_acceleration(
        df,
        id_col='id',
        frame_col='frame',
        time_col='time',
        speed_col='speed_smooth' if 'speed_smooth' in df.columns else 'speed',
        accel_col='accel_smooth',
        fps=fps
    )

    # Decompose velocity and acceleration into X and Y components
    target_speed = df['speed_smooth'] if 'speed_smooth' in df.columns else df['speed']
    df['vx'] = target_speed * np.cos(df[yaw_src])
    df['vy'] = target_speed * np.sin(df[yaw_src])
    df['ax'] = df['accel_smooth'] * np.cos(df[yaw_src])
    df['ay'] = df['accel_smooth'] * np.sin(df[yaw_src])

    # Round results for cleaner output
    for col in ['vx', 'vy', 'ax', 'ay']:
        df[col] = df[col].round(2)

    # Clean up intermediate or redundant columns
    df.drop(columns=['yaw_motion', 'accel'], inplace=True, errors='ignore')
    
    # Standardize column names
    if yaw_src == 'final_motion_yaw' and 'yaw' not in df.columns:
        df.rename(columns={'final_motion_yaw': 'yaw'}, inplace=True)
    if 'course' not in df.columns and 'heading' in df.columns:
        df.rename(columns={'heading': 'course'}, inplace=True)

    # Define desired column order for the output CSV
    ordered_cols = [
        "frame", "id", "cx", "cy", "w", "h", "r", "confidence", "type", "w_m", "h_m",
        "time", "speed", "cx_m", "cy_m", "smooth_cx", "smooth_cy", "smooth_r", "r_align",
        "isReal", "length_med", "width_med", "w_med", "h_med", "course", "yaw", "speed_smooth", "vx", "vy", "ax", "ay",
        "entry_direction", "exit_direction", "overall_direction"
    ]
    
    final_cols = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]
    return df[final_cols]


def process(traj_dir: str, output_dir: str, fps: float = 10.0, smooth: bool = True) -> None:
    """
    Batch process all trajectory CSV files in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    traj_files = [f for f in os.listdir(traj_dir) if f.endswith('.csv')]

    for traj_file in tqdm(traj_files, desc='Processing trajectory files'):
        traj_path = os.path.join(traj_dir, traj_file)
        traj_df = pd.read_csv(traj_path)
        traj_df = add_kinematic_params(traj_df, fps=fps, smooth=smooth)
        traj_df.to_csv(os.path.join(output_dir, traj_file), index=False)


if __name__ == "__main__":
    # Define input and output directories
    input_traj_dir = r"final"
    output_traj_dir = r"final"  # This will overwrite original files in 'final'
    
    process(input_traj_dir, output_traj_dir, fps=10.0, smooth=True)
    print(f"Processing complete. Results saved to: {output_traj_dir}")