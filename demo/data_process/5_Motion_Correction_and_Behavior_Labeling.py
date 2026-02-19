# -*- coding: utf-8 -*-
"""
@Time: 2025/7/4 17:59
@Auth: Terry_CYY
@File: 5_Motion_Correction_and_Behavior_Labeling.py
@IDE: PyCharm
@Function: Annotate the entry and exit orientation for motor vehicles (orientation angle is more accurate than heading angle),
            eliminate abnormal samples, and mark driving behavior (mainly based on geographical location).
"""
import numpy as np
import pandas as pd
import os
import json
import shapely
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer, CRS
from shapely.strtree import STRtree
import geopandas as gpd


# Define coordinate systems
custom_crs = "+proj=tmerc +lon_0=0 +lat_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
wgs84_crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

# Create a coordinate transformer from custom projection to WGS84
transformer = Transformer.from_crs(custom_crs, wgs84_crs, always_xy=True)


# ====================== Trajectory Direction Annotation Function (Using Position) ======================
def add_turn_directions_simple(trajectory_df, gpkg_path, max_points=30):
    """
    Rigid, no reverse driving allowed.
    Calculates the entry and exit directions of a trajectory and appends the results to the original DataFrame.
    Modification: Uses the first/last max_points for matching to reduce "Unknown" results.

    Args:
    trajectory_df: Trajectory DataFrame, containing columns ['id', 'time', 'cx_m', 'cy_m']
    gpkg_path: Path to the GeoPackage file
    max_points: Maximum number of points to check at the beginning/end (default 30)

    Returns:
    The original DataFrame with turn direction results appended.
    """
    custom_crs = "+proj=tmerc +lon_0=0 +lat_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

    # 1. Read and preprocess GeoPackage
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf[gdf['name'].str.endswith(('_in', '_out'))].copy()
    # Ensure GeoPackage data uses the custom coordinate system
    if gdf.crs is None:
        gdf = gdf.set_crs(custom_crs, allow_override=True)
    elif gdf.crs != custom_crs:
        gdf = gdf.to_crs(custom_crs)
    # Extract direction names and create subsets
    extracted = gdf['name'].str.extract(r'(.+)_(in|out)$')
    gdf['direction'] = extracted[0]  # Extract direction letter
    gdf['type'] = extracted[1]  # Extract type (in/out)

    # Separate entry and exit polygons
    entry_gdf = gdf[gdf['type'] == 'in'][['direction', 'geometry']].copy().reset_index(drop=True)
    exit_gdf = gdf[gdf['type'] == 'out'][['direction', 'geometry']].copy().reset_index(drop=True)

    # Build spatial index (global, only once)
    entry_sindex = entry_gdf.sindex
    exit_sindex = exit_gdf.sindex

    # 2. Process trajectory data - sort by id
    sorted_df = trajectory_df.sort_values(['id', 'time']).reset_index(drop=True)

    # 3. Match directions for each id (using first/last max_points)
    entry_directions = {}  # {id: entry_direction}
    exit_directions = {}  # {id: exit_direction}

    for id_, group in sorted_df.groupby('id'):
        # Take the first max_points (entry, from the beginning)
        entry_group = group.iloc[:max_points] if len(group) >= max_points else group
        entry_direction = 'Unknown'
        for _, row in entry_group.iterrows():
            pt = Point(row['cx_m'], row['cy_m'])
            candidate_idxs = list(entry_sindex.intersection(pt.bounds))
            for idx in candidate_idxs:
                poly = entry_gdf.geometry.iloc[idx]
                if pt.within(poly):
                    entry_direction = entry_gdf.at[idx, 'direction']
                    break  # Match found, stop inner loop
            if entry_direction != 'Unknown':
                break  # Match found, stop entry check for this id
        entry_directions[id_] = entry_direction

        # Take the last max_points (exit, from the end, check in reverse to prioritize the latest point)
        exit_group = group.iloc[-max_points:] if len(group) >= max_points else group
        exit_direction = 'Unknown'
        # Iterate in reverse (from the last point backwards)
        for _, row in exit_group[::-1].iterrows():
            pt = Point(row['cx_m'], row['cy_m'])
            candidate_idxs = list(exit_sindex.intersection(pt.bounds))
            for idx in candidate_idxs:
                poly = exit_gdf.geometry.iloc[idx]
                if pt.within(poly):
                    exit_direction = exit_gdf.at[idx, 'direction']
                    break
            if exit_direction != 'Unknown':
                break  # Match found, stop
        exit_directions[id_] = exit_direction

    # 4. Prepare result data
    # Create direction_results DataFrame
    ids = list(sorted_df['id'].unique())
    direction_results = pd.DataFrame({
        'id': ids,
        'entry_direction': [entry_directions.get(id_, 'Unknown') for id_ in ids],
        'exit_direction': [exit_directions.get(id_, 'Unknown') for id_ in ids]
    })
    # Generate overall turn direction
    direction_results['overall_direction'] = direction_results.apply(
        lambda row: f"{row['entry_direction']}-{row['exit_direction']}",
        axis=1
    )

    # 5. Merge results back into the original DataFrame
    result_df = trajectory_df.copy()
    result_df = result_df.merge(
        direction_results[['id', 'entry_direction', 'exit_direction', 'overall_direction']],
        on='id',
        how='left'
    )
    # Handle any possible missing values
    result_df['entry_direction'] = result_df['entry_direction'].fillna('Unknown')
    result_df['exit_direction'] = result_df['exit_direction'].fillna('Unknown')
    result_df['overall_direction'] = result_df['overall_direction'].fillna('Unknown-Unknown')

    return result_df


def add_turn_directions(trajectory_df, gpkg_path, max_points=30):
    """
    Modified direction matching logic:
    1. Allow entry to match any type of polygon (in/out)
    2. Allow exit to match any type of polygon (in/out)
    3. Prohibit matching the same direction and type
    4. Retain spatial index optimization
    """
    custom_crs = "+proj=tmerc +lon_0=0 +lat_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

    # If trajectory_df already has entry_direction and exit_direction columns, return directly
    if 'entry_direction' in trajectory_df.columns and 'exit_direction' in trajectory_df.columns:
        print("Trajectory data already contains direction information, skipping matching.")
        return trajectory_df

    # 1. Read and preprocess GeoPackage (no longer separating in/out)
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf[gdf['name'].str.endswith(('_in', '_out'))].copy()

    # Coordinate system processing
    if gdf.crs is None:
        gdf = gdf.set_crs(custom_crs, allow_override=True)
    elif gdf.crs != custom_crs:
        gdf = gdf.to_crs(custom_crs)

    # Extract direction information
    extracted = gdf['name'].str.extract(r'(.+)_(in|out)$')
    gdf['direction'] = extracted[0]  # Direction letter
    gdf['type'] = extracted[1]  # Type (in/out)

    # Build global spatial index
    full_sindex = gdf.sindex

    # 2. Process trajectory data
    sorted_df = trajectory_df.sort_values(['id', 'time']).reset_index(drop=True)

    # 3. New matching logic
    direction_results = []

    for id_, group in sorted_df.groupby('id'):
        # Entry matching (first max_points)
        entry_match = None
        for _, row in group.head(max_points).iterrows():
            pt = Point(row['cx_m'], row['cy_m'])
            candidate_idxs = list(full_sindex.intersection(pt.bounds))
            for idx in candidate_idxs:
                poly = gdf.geometry.iloc[idx]
                if pt.within(poly):
                    entry_match = {
                        'dir': gdf.at[idx, 'direction'],
                        'type': gdf.at[idx, 'type']
                    }
                    break
            if entry_match:
                break

        # Exit matching (last max_points, in reverse)
        exit_match = None
        for _, row in group.tail(max_points)[::-1].iterrows():
            pt = Point(row['cx_m'], row['cy_m'])
            candidate_idxs = list(full_sindex.intersection(pt.bounds))

            for idx in candidate_idxs:
                poly = gdf.geometry.iloc[idx]
                if pt.within(poly):
                    candidate = {
                        'dir': gdf.at[idx, 'direction'],
                        'type': gdf.at[idx, 'type']
                    }

                    # Exclude matching the same direction and type
                    if entry_match and \
                            entry_match['dir'] == candidate['dir'] and \
                            entry_match['type'] == candidate['type']:
                        continue  # Skip prohibited match

                    exit_match = candidate
                    break
            if exit_match:
                break

        # Record results
        entry_dir = entry_match['dir'] if entry_match else 'Unknown'
        exit_dir = exit_match['dir'] if exit_match else 'Unknown'
        overall = f"{entry_dir}-{exit_dir}" if entry_match or exit_match else 'Unknown'

        direction_results.append({
            'id': id_,
            'entry_direction': entry_dir,
            'exit_direction': exit_dir,
            'overall_direction': overall
        })

    # 4. Merge results
    result_df = pd.merge(
        trajectory_df,
        pd.DataFrame(direction_results),
        on='id',
        how='left'
    ).fillna({
        'entry_direction': 'Unknown',
        'exit_direction': 'Unknown',
        'overall_direction': 'Unknown'
    })

    return result_df


# ====================== Angle Calculation and Mapping Functions ======================
def angular_difference_rad(a1, a2):
    """Calculate the minimum difference between two angles (considering periodicity)"""
    diff = np.abs(a1 - a2)
    return np.minimum(diff, 2 * np.pi - diff)


def map_direction(angle, x, y, direction_type='exit'):
    """
    Map angle and position (x, y) to a direction label
    - Angle: Vehicle's direction of travel (radians), 0 is east, counter-clockwise
    - Position: Intersection center is the origin (0,0)
    - direction_type: 'entry' or 'exit'
    """
    angle = angle % (2 * np.pi)  # Normalize to [0, 2π)

    # Define angle intervals
    east_west = (angle >= 7 * np.pi / 4) or (angle < np.pi / 4)
    north_south = (np.pi / 4 <= angle < 3 * np.pi / 4)
    west_east = (3 * np.pi / 4 <= angle < 5 * np.pi / 4)
    south_north = (5 * np.pi / 4 <= angle < 7 * np.pi / 4)

    if direction_type == 'entry':
        if east_west and x < 0: return 'W'
        if north_south and y < 0: return 'S'
        if west_east and x > 0: return 'E'
        if south_north and y > 0: return 'N'

    elif direction_type == 'exit':
        if east_west and x > 0: return 'E'
        if north_south and y > 0: return 'N'
        if west_east and x < 0: return 'W'
        if south_north and y < 0: return 'S'

    return 'Unknown'


# ====================== Angle Smoothing and Correction ======================
def smooth_angle_series(angles, threshold_rad):
    """Bidirectionally smooth an angle series"""
    n = len(angles)
    if n < 2:
        return angles

    # Forward fill: find the first stable point
    first_stable_idx = n - 1
    first_stable_val = angles[-1]
    for i in range(n - 1):
        if angular_difference_rad(angles[i], angles[i + 1]) <= threshold_rad:
            first_stable_val = angles[i]
            first_stable_idx = i
            break
    angles[:first_stable_idx] = first_stable_val

    # Backward fill: find the last stable point
    last_stable_idx = 0
    last_stable_val = angles[0]
    for i in range(n - 1, 0, -1):
        if angular_difference_rad(angles[i], angles[i - 1]) <= threshold_rad:
            last_stable_val = angles[i]
            last_stable_idx = i
            break
    angles[last_stable_idx + 1:] = last_stable_val

    return angles


def correct_angles(df, angle_col, threshold_deg=60.0):
    """Apply bidirectional angle smoothing correction"""
    threshold_rad = np.deg2rad(threshold_deg)
    df_sorted = df.sort_values(['id', 'time'])

    # Create a new column name
    smooth_col = f"{angle_col}_smooth"

    # Apply smoothing by group
    df_sorted[smooth_col] = df_sorted.groupby('id')[angle_col].transform(
        lambda x: smooth_angle_series(x.values, threshold_rad)
    )
    return df_sorted


"Summary of column names generated below!!!!!"
"""
heading: Direction of motion, course (absolute bearing)
vx = speed * cos(r_align)
vy = speed * sin(r_align)
r_align: Orientation angle in the body frame (physical body direction)
yaw_motion: Corrected heading angle for motion analysis
yaw_motion_refined: Final correction
"""

# ====================== Heading, Speed, and Direction Calculation ======================
def calculate_motion_vectors(df, x_col='x', y_col='y'):
    """Calculate motion vectors and heading angle"""
    df = df.sort_values(['id', 'time'])

    # Calculate displacement vectors
    df['dx'] = df.groupby('id')[x_col].diff()
    df['dy'] = df.groupby('id')[y_col].diff()

    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate motion direction angle (heading)
        df['heading'] = np.mod(np.arctan2(df['dy'], df['dx']), 2 * np.pi)

    # Calculate time difference
    df['dt'] = df.groupby('id')['time'].diff()
    # Original speed
    df['speed'].fillna(0, inplace=True)

    return df


def calculate_body_frame_velocity(df, r_col='r_align'):
    """Calculate velocity components in the body frame"""
    # Get the raw orientation angle (physical body direction)
    r_raw = df[r_col].values

    # Global speed
    speed = df['speed'].values
    # Calculate velocity components in the body frame
    # vx: along the heading direction (tangential velocity)
    # vy: perpendicular to the heading direction (normal velocity)
    df['vx'] = speed * np.cos(r_raw)
    df['vy'] = speed * np.sin(r_raw)

    # Round to two decimal places
    df['vx'] = df['vx'].round(2)
    df['vy'] = df['vy'].round(2)

    return df


def align_yaw_with_motion(df, r_col='r_align'):
    """Correct heading angle based on motion direction (for motion analysis only)"""
    # Calculate the difference between heading and yaw
    heading = df['heading'].values
    r_raw = df[r_col].values

    # Flip the yaw angle when the difference is greater than 90 degrees
    flip_mask = np.cos(heading - r_raw) < 0
    df['yaw_motion'] = np.where(flip_mask, (r_raw + np.pi) % (2 * np.pi), r_raw)
    return df


def calculate_window_motion(group, x_col, y_col, window=10):
    """Calculate windowed displacement and direction"""
    x = group[x_col].values
    y = group[y_col].values
    time_vals = group['time'].values
    n = len(x)

    # Pre-allocate result arrays
    ds = np.full(n, np.nan)
    window_heading = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        if i - start < 1:  # Window too small
            continue

        dx = x[i] - x[start]
        dy = y[i] - y[start]

        ds[i] = np.sqrt(dx ** 2 + dy ** 2)
        if ds[i] > 0:  # Avoid division by zero
            window_heading[i] = np.arctan2(dy, dx) % (2 * np.pi)

    group['ds'] = ds
    group['window_heading'] = window_heading
    return group


def refine_motion_yaw(group, min_displacement=0.25):
    """Refine heading angle based on motion displacement (for motion analysis only)"""
    displacement = group['ds'].values
    window_heading = group['window_heading'].values
    yaw_motion = group['yaw_motion'].values.copy()

    # Only process points with valid displacement
    valid_mask = (displacement >= min_displacement) & ~np.isnan(window_heading)

    for i in np.where(valid_mask)[0]:
        diff = angular_difference_rad(window_heading[i], yaw_motion[i])
        if diff > np.pi / 2:
            yaw_motion[i] = (yaw_motion[i] + np.pi) % (2 * np.pi)

    # Set points with invalid displacement to NaN
    yaw_motion[~valid_mask] = np.nan
    group['yaw_motion_refined'] = yaw_motion
    return group


# ====================== Entry/Exit Direction Calculation ======================
def calculate_directions(df, angle_col, x_col, y_col, num_points=30):
    """Calculate the entry and exit directions for each trajectory"""
    # Ensure the angle column exists
    if angle_col not in df.columns:
        print(f"Warning: Angle column '{angle_col}' does not exist, using a fallback column")
        angle_col = 'yaw_motion_refined' if 'yaw_motion_refined' in df.columns else 'r_align'

    def get_representative_angle(angles, is_entry=True):
        """Get a representative angle"""
        n = min(num_points, len(angles))
        if n == 0:
            return np.nan
        subset = angles[:n] if is_entry else angles[-n:]
        # Calculate direction using vector averaging
        sin_sum = np.sum(np.sin(subset))
        cos_sum = np.sum(np.cos(subset))
        return np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

    def get_common_position(pos, is_entry=True):
        """Get a representative position"""
        n = min(num_points, len(pos))
        if n == 0:
            return np.nan, np.nan
        subset = pos[:n] if is_entry else pos[-n:]  # Take from the beginning or end
        return np.mean(subset)  # Average of the input column

    vehicle_types = ['car', 'bus', 'freight car', 'van', 'truck', 'pedestrian', 'moped']
    # Pre-allocate result arrays
    entry_dirs = np.empty(len(df), dtype=object)
    exit_dirs = np.empty(len(df), dtype=object)
    # Create a mapping: whether each id is a vehicle type
    id_is_vehicle = df.groupby('id')['type'].first().isin(vehicle_types).to_dict()

    # Process by group
    start_idx = 0
    for id_val, group in df.groupby('id'):
        group_size = len(group)
        end_idx = start_idx + group_size

        # Check if it is a vehicle type
        if id_is_vehicle.get(id_val, False):  # No longer subdividing vehicle types here
            angles = group[angle_col].dropna().values
            x = group[x_col].values
            y = group[y_col].values

            if len(angles) > 0:
                # Calculate entry/exit directions
                entry_angle = get_representative_angle(angles, is_entry=True)
                exit_angle = get_representative_angle(angles, is_entry=False)
                entry_x = get_common_position(x, is_entry=True)
                entry_y = get_common_position(y, is_entry=True)
                exit_x = get_common_position(x, is_entry=False)
                exit_y = get_common_position(y, is_entry=False)

                entry_dir = map_direction(entry_angle, entry_x, entry_y, 'entry')
                exit_dir = map_direction(exit_angle, exit_x, exit_y, 'exit')
            else:
                entry_dir = np.nan
                exit_dir = np.nan
        else:
            # Set to NaN for non-vehicle types
            entry_dir = np.nan
            exit_dir = np.nan

        # Fill the result arrays
        entry_dirs[start_idx:end_idx] = entry_dir
        exit_dirs[start_idx:end_idx] = exit_dir
        start_idx = end_idx
    # Check if entry/exit direction columns already exist in df
    # If not, create new columns
    if 'entry_direction' not in df.columns or 'exit_direction' not in df.columns:
        # Create new columns
        df['entry_direction'] = entry_dirs
        df['exit_direction'] = exit_dirs

        # Create overall direction column (only when both directions are not NaN)
        overall = []
        for entry, exit_ in zip(entry_dirs, exit_dirs):
            overall.append(f"{entry}-{exit_}")
        df['overall_direction'] = overall
    
    else:  # If entry/exit direction columns already exist
        # Find rows where entry and exit are "Unknown" and try to overwrite them with the results from this function
        mask_entry_unknown = df['entry_direction'] == 'Unknown'
        mask_exit_unknown = df['exit_direction'] == 'Unknown'
        # Overwrite only for rows where entry is "Unknown"
        df.loc[mask_entry_unknown, 'entry_direction'] = entry_dirs[mask_entry_unknown.values]
        # Overwrite only for rows where exit is "Unknown"
        df.loc[mask_exit_unknown, 'exit_direction'] = exit_dirs[mask_exit_unknown.values]
        # For overall_direction, also overwrite only for rows where entry or exit was "Unknown"
        mask_overall_unknown = mask_entry_unknown | mask_exit_unknown
        overall = []
        print(entry_dirs)
        for entry, exit_ in zip(entry_dirs, exit_dirs):
            overall.append(f"{entry}-{exit_}")
        df.loc[mask_overall_unknown, 'overall_direction'] = np.array(overall)[mask_overall_unknown.values]

    # Post-process results (unify format)

    return df


def modify_nan(df):
    # Replace np.nan or "Unknown" in entry_direction and exit_direction
    df['entry_direction'] = df['entry_direction'].fillna('Unknown')
    df['exit_direction'] = df['exit_direction'].fillna('Unknown')
    # Vectorized construction of overall_direction
    df['overall_direction'] = np.where(
        (df['entry_direction'] == 'Unknown') & (df['exit_direction'] == 'Unknown'),
        'Unknown-Unknown',
        np.where(
            df['entry_direction'] == 'Unknown',
            'Unknown-' + df['exit_direction'],
            np.where(
                df['exit_direction'] == 'Unknown',
                df['entry_direction'] + '-Unknown',
                df['entry_direction'] + '-' + df['exit_direction']
            )
        )
    )

    return df


# ====================== Main Processing Flow ======================
def process_trajectory_data(input_path, output_path, gpkg_path, type_keep=None):
    """Complete workflow for processing trajectory data"""
    # 1. Read data
    df = pd.read_csv(input_path)

    # if type_keep:
    #     df = df[df['type'].isin(type_keep)]

    # 2. Calculate motion vectors, speed, and heading
    df = calculate_motion_vectors(df, x_col='cx_m', y_col='cy_m')

    # 3. Calculate body frame velocity components (using raw orientation angle)
    df = calculate_body_frame_velocity(df, r_col='r_align')

    # 4. Create orientation angle for motion analysis (does not affect original physical orientation)
    df = align_yaw_with_motion(df, r_col='r_align')

    # 5. Parallel computation of windowed motion
    groups = df.groupby('id')
    processed = Parallel(n_jobs=-1)(
        delayed(calculate_window_motion)(group, 'cx_m', 'cy_m')
        for _, group in groups
    )
    df = pd.concat(processed)

    # 6. Further refine motion analysis orientation angle based on windowed motion
    corrected = Parallel(n_jobs=-1)(
        delayed(refine_motion_yaw)(group)
        for _, group in df.groupby('id')
    )
    df = pd.concat(corrected)

    # 7. Angle smoothing - motion analysis orientation angle
    df = correct_angles(df, angle_col='yaw_motion_refined', threshold_deg=60.0)


    # Initially get a flow direction based on position
    df = add_turn_directions(df, gpkg_path)

    # # 8. Calculate entry/exit directions - using smoothed motion analysis orientation angle (to supplement incorrect flow directions)
    smooth_col = 'yaw_motion_refined_smooth'
    x_col = 'cx_m'
    y_col = 'cy_m'
    df = calculate_directions(df, angle_col=smooth_col, x_col=x_col, y_col=y_col)

    # 9. Clean up and save results
    df = df.sort_values(['id', 'time'])

    # Round angle columns
    angle_cols = ['heading', 'r_align', 'yaw_motion', 'yaw_motion_refined', smooth_col]
    for col in angle_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Create final motion analysis angle column
    df['final_motion_yaw'] = df.get(smooth_col,
                                    df.get('yaw_motion_refined',
                                           df.get('yaw_motion', np.nan)))

    # Remove unnecessary output columns
    df = df.drop(columns=['dx', 'dy', 'dt', 'ds', 'window_heading', 'yaw_motion_refined', 'yaw_motion_refined_smooth'],
                 errors='ignore')

    df = modify_nan(df)

    df.to_csv(output_path, index=False)

    return df


# ====================== Example Usage ======================
if __name__ == "__main__":
    # Configuration parameters
    INPUT_DIR = r"final"  # Your root directory
    pattern = os.path.join(INPUT_DIR, "**", "*.csv")
    csv_files = glob(pattern, recursive=True)

    edge_file = r"maps/FIDRT_edges.gpkg"

    for csv_file in tqdm(csv_files, desc="Batch processing CSV files"):
        process_trajectory_data(
            input_path=csv_file,
            output_path=csv_file,  # Overwrite original file
            gpkg_path=edge_file,
        )
    print("All processing complete!")