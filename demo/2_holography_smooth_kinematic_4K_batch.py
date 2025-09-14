# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/4 1:37
@Author  : Terry CYY
@FileName: 2_holography_smooth_kinematic_4K_batch.py
@Software: PyCharm
@Function: Batch convert the pixel coordinates of targets in 4K video to latitude and longitude coordinates (smooth first, then interpolate, using kinematic interpolation - final version)
"""

import pandas as pd
import numpy as np
import cv2
from math import radians, cos, sin, atan2, sqrt, pi, degrees
from pyproj import Proj
import os
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import Tuple, Literal, Dict
from scipy.interpolate import PchipInterpolator


# Savitzky-Golay filter (S-G filter) configuration class
@dataclass
class SGConfig:
    # Index columns
    id_col: str = "id"
    frame_col: str = "frame"  # or 'time'

    # Fields to be smoothed
    scalar_cols: Tuple[str, ...] = ("cx", "cy")  # Ordinary real numbers
    angle_cols: Tuple[str, ...] = ("r",)  # Angle, period π

    # S-G parameters
    window_len: int = 7  # Must be odd
    polyorder: int = 3
    mode: Literal["interp", "nearest", "mirror", "constant"] = "interp"


# ---------- Helper functions ----------
def _best_window(n_points: int, cfg: SGConfig) -> int:  # Safe window
    """
    Returns an odd window that satisfies the conditions:
    1) ≤ n_points
    2) ≥ polyorder + 1
    If not possible, returns 0 to indicate "no filtering".
    """
    # Trajectory is shorter than polyorder+1 ⇒ give up directly
    if n_points <= cfg.polyorder:
        return 0

    # First try to use the user-defined window_len, and crop it to within n_points
    w = min(cfg.window_len, n_points)
    if w % 2 == 0:                      # Adjust to odd
        w -= 1
    # If it is still ≤ polyorder, adjust the window to n_points or n_points-1 (keep it odd)
    if w <= cfg.polyorder:
        w = n_points if n_points % 2 else n_points - 1
    # Final check
    if w <= cfg.polyorder:
        return 0
    return w


def _sg(arr: np.ndarray, win: int, cfg: SGConfig) -> np.ndarray:
    """
    Safe Savitzky-Golay call:
    win == 0 or win <= polyorder ⇒ return original value directly
    """
    if win == 0 or win <= cfg.polyorder or len(arr) < win:
        return arr
    return savgol_filter(arr, win, cfg.polyorder, mode=cfg.mode)


def _sg_angle(arr: np.ndarray, win: int, cfg: SGConfig) -> np.ndarray:
    if len(arr) <= cfg.polyorder:
        return arr
    c, s = np.cos(2 * arr), np.sin(2 * arr)
    c_s, s_s = _sg(c, win, cfg), _sg(s, win, cfg)
    return (np.arctan2(s_s, c_s) / 2) % np.pi  # 0~π


# ---------- Main function ----------
def sg_filter_bbox(df_in: pd.DataFrame,
                   cfg: SGConfig = SGConfig(),
                   out_map: dict | None = None,  # <–– New, map results to new columns
                   keep_orig: bool = True):
    """
    General Savitzky-Golay smoothing: supports both OBB / HBB
    """
    df = df_in.sort_values([cfg.id_col, cfg.frame_col]).reset_index(drop=True)
    scalar_cols = [c for c in cfg.scalar_cols if c in df.columns]
    angle_cols = [c for c in cfg.angle_cols if c in df.columns]
    # Pre-allocate storage for results
    result_store = {col: np.empty(len(df)) for col in scalar_cols + angle_cols}

    # Group by id
    for _, idx in df.groupby(cfg.id_col, sort=False).indices.items():
        gidx = np.fromiter(idx, dtype=int)
        npt = len(gidx)
        win = _best_window(npt, cfg)

        # Scalar columns
        for col in scalar_cols:
            arr = df.loc[gidx, col].to_numpy()
            result_store[col][gidx] = _sg(arr, win, cfg)

        # Angle columns
        for col in angle_cols:
            arr = df.loc[gidx, col].to_numpy()
            result_store[col][gidx] = _sg_angle(arr, win, cfg)

    # ==== Write back ====
    for col, arr in result_store.items():
        if out_map and col in out_map:  # Explicitly specified target column
            target = out_map[col]
        else:  # If not specified, handle according to keep_orig / suffix
            target = (col + '_sg') if keep_orig else col
        df[target] = arr

    return df


# ---------- Position and angle interpolation methods ----------

def _interp_angle(frames, r_vals, full_frames):
    """0~π angle PCHIP interpolation to avoid jumps across π"""
    r_vals = np.asarray(r_vals)  # Ensure it is an ndarray
    z = np.exp(1j * 2 * r_vals)  # onto unit circle
    re = PchipInterpolator(frames, z.real)(full_frames)
    im = PchipInterpolator(frames, z.imag)(full_frames)
    return (np.angle(re + 1j * im) / 2) % np.pi


# ---------- Method 1: PCHIP --------------------------------
def _fill_pchip(grp, cfg):
    g = grp.sort_values(cfg['frame']).set_index(cfg['frame'])
    full_frames = np.arange(g.index.min(), g.index.max() + 1)
    g_full = g.reindex(full_frames)

    # Mark if original
    g_full['isReal'] = np.where(g_full[cfg['id']].notna(), 1, 0)

    # Constant columns forward/backward fill
    for col in (cfg['w_med'], cfg['h_med'], cfg['type']):
        g_full[col] = g_full[col].ffill().bfill()

    # --- Position interpolation ---
    mask = g_full['isReal'].astype(bool).values  # Real frame index
    if mask.sum() > 1:  # At least 2 points are needed for PCHIP
        frames_known = full_frames[mask]
        for col in (cfg['sx'], cfg['sy']):
            g_full[col] = PchipInterpolator(frames_known,
                                            g_full.loc[frames_known, col])(full_frames)
        # Yaw interpolation
        if cfg['sr'] in g_full.columns:
            r_vals = g_full.loc[frames_known, cfg['sr']]
            g_full[cfg['sr']] = _interp_angle(frames_known, r_vals, full_frames)

    g_full[cfg['id']] = g_full[cfg['id']].ffill().bfill()
    # Keep decimals
    g_full[cfg['sx']] = g_full[cfg['sx']].round(2)
    g_full[cfg['sy']] = g_full[cfg['sy']].round(2)
    if cfg['sr'] in g_full.columns:
        g_full[cfg['sr']] = g_full[cfg['sr']].round(4)  # Keep four decimal places for radians

    return g_full.reset_index().rename(columns={'index': cfg['frame']})


# ------------------------------------------------------------------
# Kinematic interpolation
# ------------------------------------------------------------------
def _kinematic_interpolation(
        frames: np.ndarray,  # Known frame numbers
        positions: np.ndarray,  # Known positions Nx2
        angles: np.ndarray,  # Known angles (radians, 0~π)
        full_frames: np.ndarray  # Complete frame sequence
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate position & angle based on kinematic constraints (with head/tail extrapolation + linear interpolation as a fallback)"""

    # ---------- 0. Result initialization ----------
    interp_positions = np.full((len(full_frames), 2), np.nan, dtype=np.float64)
    interp_angles = np.full(len(full_frames), np.nan, dtype=np.float64)
    # Create a mapping from frame number to index
    frame2idx = {f: i for i, f in enumerate(full_frames)}
    # Fill in known frame data
    for p, a, f in zip(positions, angles, frames):
        idx = frame2idx[f]
        interp_positions[idx] = p
        interp_angles[idx] = a
    # Mark known frame positions
    known_mask = np.isin(full_frames, frames)
    known_indices = np.where(known_mask)[0]
    # Handle single or zero frame cases
    if len(known_indices) <= 1:
        # Linear interpolation of position
        for k in range(2):
            s = pd.Series(interp_positions[:, k])
            s = s.interpolate(method='linear', limit_direction='both')
            interp_positions[:, k] = s.values
        # Angle filling
        s = pd.Series(interp_angles)
        s = s.interpolate(method='nearest', limit_direction='both')
        interp_angles = s.values
        return interp_positions, interp_angles

    # ---------- 1. Angle difference calculation function (to prevent reversal) ----------
    def calculate_angle_diff(start_ang, end_ang):
        """Calculate the minimum angle difference in the range 0~π (to avoid direction reversal)"""
        diff = end_ang - start_ang
        # Handle the case where the angle crosses the π boundary
        if diff > np.pi / 2:
            return diff - np.pi  # Counterclockwise across the boundary
        elif diff < -np.pi / 2:
            return diff + np.pi  # Clockwise across the boundary
        return diff
    # ---------- 2. Divide into segments by "two adjacent known frames" ----------
    segments = []  # Each segment only stores the index range of unknown
    # Leading missing frames (from 0 to before the first known frame)
    if known_indices[0] > 0:
        segments.append((0, known_indices[0] - 1))
    # Intermediate missing frames (between adjacent known frames)
    for i in range(len(known_indices) - 1):
        left = known_indices[i]
        right = known_indices[i + 1]
        if right - left > 1:
            segments.append((left + 1, right - 1))
    # Trailing missing frames (after the last known frame)
    if known_indices[-1] < len(full_frames) - 1:
        segments.append((known_indices[-1] + 1, len(full_frames) - 1))

    # ---------- 3. Calculate dynamic displacement threshold ----------
    if len(positions) > 2:
        # Calculate the median distance between known positions
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        min_dist = np.median(displacements) * 0.01  # Automatic scaling
    else:
        min_dist = 1e-6  # Default threshold

    # ---------- 4. Interpolate / extrapolate segment by segment ----------
    for seg_start, seg_end in segments:
        # Get boundary indices
        left_known_idx = seg_start - 1 if seg_start > 0 else None
        right_known_idx = seg_end + 1 if seg_end < len(full_frames) - 1 else None
        # Index range to be interpolated
        seg_indices = np.arange(seg_start, seg_end + 1)
        # Trailing extrapolation (only left known point)
        if right_known_idx is None:
            # Get known frame indices that can be used to calculate speed
            speed_indices = [i for i in range(left_known_idx - 2, left_known_idx + 1)
                             if i >= 0 and known_mask[i]]
            # Calculate weighted average speed (the closer the higher the weight)
            if len(speed_indices) > 1:
                weights = np.arange(1, len(speed_indices))
                total_weight = weights.sum()
                v = np.zeros(2)
                for i in range(1, len(speed_indices)):
                    w = weights[i - 1] / total_weight
                    delta = interp_positions[speed_indices[i]] - interp_positions[speed_indices[i - 1]]
                    v += w * delta
            else:
                v = np.array([0.0, 0.0])
            # Vectorized extrapolation calculation
            steps = np.arange(1, len(seg_indices) + 1)
            interp_positions[seg_indices] = interp_positions[left_known_idx] + steps[:, None] * v
            interp_angles[seg_indices] = interp_angles[left_known_idx]

        # Leading extrapolation (only right known point)
        elif left_known_idx is None:
            # Get known frame indices that can be used to calculate speed
            speed_indices = [i for i in range(right_known_idx, right_known_idx + 3)
                             if i < len(full_frames) and known_mask[i]]
            # Calculate weighted average speed (the closer the higher the weight)
            if len(speed_indices) > 1:
                weights = np.arange(1, len(speed_indices))
                total_weight = weights.sum()
                v = np.zeros(2)

                for i in range(len(speed_indices) - 1):
                    w = weights[-(i + 1)] / total_weight  # Reverse weight
                delta = interp_positions[speed_indices[i + 1]] - interp_positions[speed_indices[i]]
                v += w * delta
            else:
                v = np.array([0.0, 0.0])
            # Vectorized extrapolation calculation
            steps = np.arange(len(seg_indices), 0, -1)
            interp_positions[seg_indices] = interp_positions[right_known_idx] - steps[:, None] * v
            interp_angles[seg_indices] = interp_angles[right_known_idx]

        # Intermediate interpolation (known points on both sides)
        else:
            # Get boundary values
            start_pos = interp_positions[left_known_idx]
            end_pos = interp_positions[right_known_idx]
            start_angle = interp_angles[left_known_idx]
            end_angle = interp_angles[right_known_idx]
            # Calculate displacement and number of frames
            displacement = end_pos - start_pos
            total_frames = right_known_idx - left_known_idx
            # Calculate interpolation ratio
            alpha = (seg_indices - left_known_idx) / total_frames
            # Position interpolation (linear)
            interp_positions[seg_indices] = start_pos + alpha[:, None] * displacement
            # Angle interpolation (anti-reversal)
            angle_diff = calculate_angle_diff(start_angle, end_angle)
            interpolated_angles = start_angle + alpha * angle_diff
            # Normalize angle to [0, π) range
            interp_angles[seg_indices] = interpolated_angles % np.pi

        # ---------- 5. Fallback to handle remaining NaNs ----------
        # Linear interpolation of position
        for k in range(2):
            s = pd.Series(interp_positions[:, k])
            s = s.interpolate(method='linear', limit_direction='both')
            interp_positions[:, k] = s.values
        # Nearest neighbor interpolation of angle
        s = pd.Series(interp_angles)
        s = s.interpolate(method='nearest', limit_direction='both')
        interp_angles = s.values

    return interp_positions, interp_angles


# ------------------------------------------------------------------
# group level filling
# ------------------------------------------------------------------
def _fill_with_kinematics(grp: pd.DataFrame, cfg: Dict[str, str]) -> pd.DataFrame:
    """Fill missing frames for trajectories with the same id"""
    g = grp.sort_values(cfg['frame']).reset_index(drop=True)

    # ---------- 1. Generate complete frame column ----------
    full_frames = np.arange(g[cfg['frame']].min(), g[cfg['frame']].max() + 1)
    g_full = pd.DataFrame({cfg['frame']: full_frames})

    # ---------- 2. Merge & mark ----------
    g_full = g_full.merge(g, on=cfg['frame'], how='left')
    g_full['isReal'] = (g_full[cfg['id']].notna()).astype(int)

    # Constant columns do ffill/bfill (fill directly)
    for key in ['w_med', 'h_med', 'type', 'w_corr', 'h_corr', 'w', 'h']:
        # First determine if this key exists in cfg
        if key in cfg:
            col = cfg[key]
            # Then determine if this column exists in g_full
            if col in g_full:
                # Interpolation methods can be used here
                g_full[col] = g_full[col].ffill().bfill()

    # ---------- 3. Get known frame information ----------
    known_mask = g_full['isReal'] == 1
    known_frames = g_full.loc[known_mask, cfg['frame']].values
    known_pos = g_full.loc[known_mask, [cfg['sx'], cfg['sy']]].values
    if cfg['sr'] in g_full:
        known_ang = g_full.loc[known_mask, cfg['sr']].fillna(0.).values
    else:
        known_ang = np.zeros(len(known_frames))

    # ---------- 4. Kinematic interpolation ----------
    interp_pos, interp_ang = _kinematic_interpolation(
        known_frames, known_pos, known_ang, full_frames
    )

    # Write back
    g_full[cfg['sx']] = interp_pos[:, 0]
    g_full[cfg['sy']] = interp_pos[:, 1]
    if cfg['sr'] in g_full:
        g_full[cfg['sr']] = interp_ang

    # Keep original values for original frames (to prevent floating point errors)
    g_full.loc[known_mask, [cfg['sx'], cfg['sy']]] = known_pos
    if cfg['sr'] in g_full:
        g_full.loc[known_mask, cfg['sr']] = known_ang

    # Fill id
    g_full[cfg['id']] = g_full[cfg['id']].ffill().bfill()
    return g_full


# Write directly in the function

def fill_missing_frames(
        df: pd.DataFrame,
        id_col: str = 'id',
        frame_col: str = 'frame',
        w_col: str = 'w',
        h_col: str = 'h',
        w_corr_col: str = 'w_corr',
        h_corr_col: str = 'h_corr',
        sx_col: str = 'smooth_cx',
        sy_col: str = 'smooth_cy',
        sr_col: str = 'smooth_r',
        wmed_col: str = 'w_med',
        hmed_col: str = 'h_med',
        type_col: str = 'type'
) -> pd.DataFrame:
    """
    Fill missing frames using kinematic constraints + linear fallback
    """
    # If w_med / h_med already exists, you can skip it
    if wmed_col not in df:
        df[wmed_col] = df.groupby(id_col)['w'].transform('median').round(2)
    if hmed_col not in df:
        df[hmed_col] = df.groupby(id_col)['h'].transform('median').round(2)

    cfg = dict(id=id_col, frame=frame_col, sx=sx_col, sy=sy_col, sr=sr_col,
                w=w_col, h=h_col, w_corr=w_corr_col, h_corr=h_corr_col,
               w_med=wmed_col, h_med=hmed_col, type=type_col)

    results = []
    for _, grp in df.groupby(id_col, sort=False):
        results.append(_fill_with_kinematics(grp, cfg))

    # sx_col, sy_col only keep the first two digits
    df[sx_col] = df[sx_col].round(2)
    df[sy_col] = df[sy_col].round(2)
    if sr_col in df:
        df[sr_col] = df[sr_col].round(4)  # Keep four decimal places for radians

    return pd.concat(results, ignore_index=True)


# Coordinate transformation related functions
def image_to_gps(x, y, H):
    """Use homography matrix to convert pixel coordinates to longitude and latitude"""
    # Create homogeneous coordinates
    point = np.array([[x], [y], [1.0]])

    # Apply homography transformation
    transformed = np.dot(H, point)

    # Homogeneous coordinate normalization
    lon = transformed[0, 0] / transformed[2, 0]
    lat = transformed[1, 0] / transformed[2, 0]

    return lon, lat


def calculate_homography(ref_df):
    """Calculate the homography matrix from image coordinates to geographic coordinates"""
    # Image coordinate points (pixels)
    src_points = ref_df[['x', 'y']].values.astype(np.float32)
    print(src_points)

    # Geographic coordinate points (longitude and latitude)
    dst_points = ref_df[['lon', 'lat']].values.astype(np.float32)

    # Calculate homography matrix
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    print("Homography matrix calculation completed")
    print(f"Inlier ratio: {np.sum(mask) / len(mask):.2%}")
    print(f"Homography matrix:\n{H}")

    return H


def calculate_meters_per_pixel(ref_df):
    """Calculate the average meters/pixel ratio"""
    distances = []

    for i in range(len(ref_df)):
        for j in range(i + 1, len(ref_df)):
            p1 = ref_df.iloc[i]
            p2 = ref_df.iloc[j]

            # Calculate image distance (pixels)
            dx_img = p2['x'] - p1['x']
            dy_img = p2['y'] - p1['y']
            img_dist_px = sqrt(dx_img ** 2 + dy_img ** 2)

            # Calculate geographic distance (meters) - using Haversine formula
            lon1, lat1 = radians(p1['lon']), radians(p1['lat'])
            lon2, lat2 = radians(p2['lon']), radians(p2['lat'])

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            geo_dist_m = 6371000 * c

            if img_dist_px > 0:
                distances.append(geo_dist_m / img_dist_px)

    # Calculate average meters/pixel ratio
    return np.mean(distances)


def calculate_rotation_angle(ref_df):
    """Calculate the rotation angle between coordinate systems"""
    # Select the two farthest points
    max_dist = 0
    p1_idx, p2_idx = 0, 1

    for i in range(len(ref_df)):
        for j in range(i + 1, len(ref_df)):
            dx = ref_df.iloc[j]['x'] - ref_df.iloc[i]['x']
            dy = ref_df.iloc[j]['y'] - ref_df.iloc[i]['y']
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist > max_dist:
                max_dist = dist
                p1_idx, p2_idx = i, j

    p1 = ref_df.iloc[p1_idx]
    p2 = ref_df.iloc[p2_idx]
    # print('Reference points taken:', p1, p2)

    # Vector in image coordinate system
    img_vec = np.array([p2['x'] - p1['x'], p2['y'] - p1['y']])

    # Vector in geographic coordinate system (east direction, north direction)
    # Use UTM coordinates to calculate geographic vectors
    utm_proj = Proj(proj='utm', zone=50, ellps='WGS84')
    x1, y1 = utm_proj(p1['lon'], p1['lat'])
    x2, y2 = utm_proj(p2['lon'], p2['lat'])
    geo_vec = np.array([x2 - x1, y2 - y1])

    # Calculate angle difference
    img_angle = atan2(-img_vec[1], img_vec[0])
    geo_angle = atan2(geo_vec[1], geo_vec[0])

    print(img_angle, geo_angle)

    rotation_angle = geo_angle - img_angle

    # Normalize angle
    rotation_angle = atan2(sin(rotation_angle), cos(rotation_angle))

    print(f"Detected coordinate system rotation angle: {degrees(rotation_angle):.2f}°")
    return rotation_angle


def calculate_speed(group):
    group = group.sort_values('time')
    # Use UTM coordinates to calculate displacement
    x = group['utm_x_smooth'].values
    y = group['utm_y_smooth'].values
    t = group['time'].values
    # Difference between adjacent points
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    # Calculate displacement (unit: meters)
    displacement = np.sqrt(dx ** 2 + dy ** 2)
    # Calculate speed (unit: m/s)
    speed = displacement / dt
    # Extend the result, the first frame has no previous frame, set to NaN
    speed = np.concatenate([[np.nan], speed])
    return pd.Series(speed, index=group.index, name='speed')


def process(input_path, output_path, cal_path, filename, utm_center, fps=10.0):
    # ----------------------------
    # 1. Configuration: filename & constants
    # ----------------------------
    # ----------------------------
    if filename.endswith('.csv'):
        # print(filename)
        # Split the filename before "_yolo" as the reference point filename
        refname = filename.split('_yolo')[0]
        print('Processing video:', refname)
        # Find the csv file containing the refname string in cal_path and save the path to a list
        ref_csvs = [os.path.join(cal_path, f) for f in os.listdir(cal_path) if refname in f and f.endswith('.csv')]
        # Find the csv file containing the filename string in input_path and save the path to a list
        track_csvs = [os.path.join(input_path, f) for f in os.listdir(input_path) if filename in f and f.endswith('.csv')]

        for ref_csv, track_csv in zip(ref_csvs, track_csvs):
            # ----------------------------
            # 2. Read reference points
            # ----------------------------
            ref_df = pd.read_csv(ref_csv)
            ref_df.columns = ['point', 'lon', 'lat', 'x', 'y']

            # ----------------------------
            # 3. Homography matrix transformation
            # ----------------------------
            # Calculate homography matrix
            H = calculate_homography(ref_df)

            # ----------------------------
            # 4. Calculate the scaling factor after reference point transformation
            # ----------------------------
            meters_per_pixel = calculate_meters_per_pixel(ref_df)
            print(f"Average scaling factor: {meters_per_pixel:.4f} meters/pixel")

            # ----------------------------
            # 5. Read trajectory data and convert
            # ----------------------------
            # Extract base image name
            if 'hbb' in track_csv:
                is_obb = False
            elif 'obb' in track_csv:
                is_obb = True
            else:
                print(f"Warning: {track_csv} does not contain 'HBB' or 'OBB' identifier, skip")
                continue

            track_df = pd.read_csv(track_csv)

            # Common configuration parameters
            common_config = {
                'id_col': 'id',
                'frame_col': 'frame',
                'window_len': 10,
                'polyorder': 3,
                'mode': 'interp',
            }
            # Configure SG filter according to bbox type
            if not is_obb:  # HBB processing
                sg_config = SGConfig(
                    scalar_cols=('cx', 'cy'),
                    **common_config
                )
                out_map = {'cx': 'smooth_cx', 'cy': 'smooth_cy'}
            else:  # OBB processing
                sg_config = SGConfig(
                    scalar_cols=('cx', 'cy'),
                    angle_cols=('r',),
                    **common_config
                )
                out_map = {'cx': 'smooth_cx', 'cy': 'smooth_cy', 'r': 'smooth_r'}
            # Apply Savitzky-Golay filter
            track_df = sg_filter_bbox(track_df, cfg=sg_config, out_map=out_map)
            print(track_df.columns)

            track_df = fill_missing_frames(track_df)  # Smoothing

            # First simply calculate the time from the frame (10 frames per second)
            track_df['time'] = track_df['frame'] / fps  # The frame rate is 10 frames/second, and the original frame rate is 29.97 frames/second
            # Keep only the first two decimal places
            track_df['time'] = track_df['time'].round(2)

            center_lons, center_lats = [], []
            W_m_list, H_m_list = [], []
            adjusted_r_list = []  # Store the adjusted orientation angle

            # ----------------------------
            # 6. Calculate rotation angle
            # ----------------------------

            rotation_angle = calculate_rotation_angle(ref_df)

            # ----------------------------
            # 7. Traverse each record to calculate the center point longitude and latitude, physical size and adjusted orientation angle
            # ----------------------------
            for _, row in track_df.iterrows():
                cx, cy, w_px, h_px = row['smooth_cx'], row['smooth_cy'], row['w'], row['h']

                # 7.1 Calculate the center point longitude and latitude
                lon, lat = image_to_gps(cx, cy, H)
                center_lons.append(round(lon, 7))
                center_lats.append(round(lat, 7))

                # 7.2 Calculate the actual physical size of the object (meters)
                W_m = w_px * meters_per_pixel
                H_m = h_px * meters_per_pixel
                W_m_list.append(round(W_m, 2))
                H_m_list.append(round(H_m, 2))

                # 7.3 Adjust the target orientation angle (if the r column exists)
                if 'r' in row:
                    original_r = row['smooth_r']
                    # Adjust the orientation angle: original angle + rotation angle (first take the opposite, normalize, and then add the rotation angle)
                    adjusted_r = (-original_r) % (2 * np.pi) + rotation_angle

                    # Normalize the angle to the range [0, π)
                    adjusted_r = adjusted_r % (2 * pi)
                    if adjusted_r < 0:
                        adjusted_r += 2 * pi
                    # adjusted_r = adjusted_r % pi  # Symmetric direction

                    adjusted_r_list.append(round(adjusted_r, 4))
                else:
                    # If there is no r column in the data, add a placeholder value
                    adjusted_r_list.append(None)

            # ----------------------------
            # 8. Write back to DataFrame and save
            # ----------------------------
            track_df['center_lons'] = center_lons
            track_df['center_lats'] = center_lats
            track_df['w_m'] = W_m_list  # Change to lowercase
            track_df['h_m'] = H_m_list

            # If there is an r column in the original data, add the adjusted orientation angle
            if 'r' in track_df.columns:
                track_df['adjusted_r'] = adjusted_r_list

            # ----------------------------
            # 9. Calculate speed
            # ----------------------------

            # Use UTM Zone 50
            utm_proj = Proj(proj='utm', zone=50, ellps='WGS84')
            utm_coords = utm_proj(track_df['center_lons'].values, track_df['center_lats'].values)
            track_df['utm_x_smooth'] = utm_coords[0]
            track_df['utm_y_smooth'] = utm_coords[1]

            # Apply the function and process the results
            track_df['speed'] = track_df.groupby('id').apply(calculate_speed).reset_index(drop=True)

            # Keep the first two digits
            # smooth_x, smooth_y, smooth_r (if any)
            track_df['smooth_cx'] = track_df['smooth_cx'].round(2)
            track_df['smooth_cy'] = track_df['smooth_cy'].round(2)
            if 'smooth_r' in track_df.columns:
                track_df['smooth_r'] = track_df['smooth_r'].round(4)
            track_df['speed'] = track_df['speed'].round(2)
            track_df['utm_x_smooth'] = track_df['utm_x_smooth'].round(2)
            track_df['utm_y_smooth'] = track_df['utm_y_smooth'].round(2)

            # Convert to relative coordinates
            # Check if utm_x_smooth and utm_y_smooth columns exist
            if 'utm_x_smooth' in track_df.columns and 'utm_y_smooth' in track_df.columns:
                print('UTM column exists...')
                # Calculate relative coordinates
                track_df['cx_m'] = track_df['utm_x_smooth'] - utm_center[0]  # Horizontal coordinate
                track_df['cy_m'] = track_df['utm_y_smooth'] - utm_center[1]  # Vertical coordinate
                # Keep only two decimal places
                track_df['cx_m'] = track_df['cx_m'].round(2)
                track_df['cy_m'] = track_df['cy_m'].round(2)
                # Delete the original utm_x_smooth and utm_y_smooth columns
                track_df.drop(columns=['utm_x_smooth', 'utm_y_smooth'], inplace=True)

            out_file = os.path.join(output_path, filename)
            track_df.to_csv(out_file, index=False)
            print(f"Done: Results saved to {out_file}")


if __name__ == '__main__':
    # The detailed information of UTM is hidden here, and the horizontal and vertical coordinates of UTM50N are fixed to the lower left corner of the area.
    utm_center = (0, 0)  # Privacy protection

    inputPath = r"input"
    calPath = r'calibration'
    outPath = r'output'
    for name in os.listdir(inputPath):
        process(inputPath, outPath, calPath, name, utm_center, fps=10)  # fps is adjustable