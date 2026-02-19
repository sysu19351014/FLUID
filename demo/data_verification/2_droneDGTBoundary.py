# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/31 12:29
@Author  : Terry_CYY
@File    : 2_droneDGTBoundary.py
@IDE     : PyCharm
@Function: Recalculate Post-Encroachment Time / Dynamic Gap Time (PET/DGT) based on bounding box 
           intersections for drone trajectory datasets.
"""
import os
import numpy as np
import pandas as pd
import argparse
from rtree import index
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Default Configurations ---
DEFAULT_INPUT_DIR = r"result"
DEFAULT_OUTPUT_DIR = r"conflict"

# Argument Parsing
parser = argparse.ArgumentParser(description="Calculate PET based on bounding box intersections.")
parser.add_argument('--input_dir', type=str, help='Path to the input CSV directory')
parser.add_argument('--output_dir', type=str, help='Path to the output CSV directory')
args = parser.parse_args()

input_dir = args.input_dir if args.input_dir else DEFAULT_INPUT_DIR
output_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path, id_col, time_col, x_col, y_col, angle_col='angle', speed_col='speed',
              length_col='length', width_col='width', type_col='type', drop_types=None):
    """
    Load trajectory data including position, angle, speed, and vehicle dimensions.
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=[id_col, time_col]).reset_index(drop=True)

    # Filter out excluded vehicle types
    if drop_types is not None and type_col in df.columns:
        df = df[~df[type_col].isin(drop_types)]

    # Calculate heading vectors (assuming angle is in radians)
    df['hx'] = np.cos(df[angle_col])
    df['hy'] = np.sin(df[angle_col])

    # Build trajectory data structures
    trajectories = {}
    times = {}
    vehicle_data = {}

    for vid in df[id_col].unique():
        vdf = df[df[id_col] == vid]
        trajectories[vid] = np.column_stack((vdf[x_col], vdf[y_col]))
        times[vid] = vdf[time_col].values.astype(float)
        vehicle_data[vid] = {
            'hx': vdf['hx'].values,
            'hy': vdf['hy'].values,
            'speed': vdf[speed_col].values if speed_col in df.columns else np.zeros(len(vdf)),
            'length': vdf[length_col].values[0] if length_col in df.columns else 4.5,
            'width': vdf[width_col].values[0] if width_col in df.columns else 2.0,
            'angle': vdf[angle_col].values  # Store original angle values
        }

    return trajectories, times, vehicle_data, df


def get_bbox_corners(x, y, hx, hy, length, width):
    """
    Calculate the coordinates of the four corners of a vehicle's bounding box.
    """
    # Normalize heading vector
    h_norm = np.sqrt(hx ** 2 + hy ** 2)
    if h_norm == 0:
        h_norm = 1
    hx_norm = hx / h_norm
    hy_norm = hy / h_norm
    
    # Perpendicular vector for width
    perp_hx = -hy_norm
    perp_hy = hx_norm
    
    # Centers of front and rear edges
    front_center = np.array([x + hx_norm * length / 2, y + hy_norm * length / 2])
    rear_center = np.array([x - hx_norm * length / 2, y - hy_norm * length / 2])
    
    # Four corner points
    corners = np.array([
        front_center + np.array([perp_hx, perp_hy]) * width / 2,  # Front-Right
        front_center - np.array([perp_hx, perp_hy]) * width / 2,  # Front-Left
        rear_center - np.array([perp_hx, perp_hy]) * width / 2,   # Rear-Left
        rear_center + np.array([perp_hx, perp_hy]) * width / 2    # Rear-Right
    ])

    return corners


def check_bbox_collision(corners1, corners2):
    """
    Check if two bounding boxes intersect using the Separating Axis Theorem (SAT).
    """
    def get_axes(corners):
        """Get the two primary axes vectors of the bounding box."""
        axes = []
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            # Get axis perpendicular to the edge
            axis = np.array([-edge[1], edge[0]])
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
                axes.append(axis)
        return axes[:2]  # Rectangles only need two axes

    def project_onto_axis(corners, axis):
        """Project corner points onto a specific axis."""
        projections = np.dot(corners, axis)
        return projections.min(), projections.max()

    # Retrieve axes from both bounding boxes
    axes1 = get_axes(corners1)
    axes2 = get_axes(corners2)
    
    # Check for gaps on all axes
    for axis in axes1 + axes2:
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)
        # If there is no overlap on even one axis, the boxes do not intersect
        if max1 < min2 or max2 < min1:
            return False
    return True


def build_rtree(trajectories):
    """
    Construct an R-Tree index for trajectory bounding boxes.
    """
    idx = index.Index()
    for i, (traj_id, traj_points) in enumerate(trajectories.items()):
        min_x, min_y = np.min(traj_points, axis=0)
        max_x, max_y = np.max(traj_points, axis=0)
        idx.insert(i, (min_x, min_y, max_x, max_y), obj=traj_id)
    return idx


def potential_pairs(trajectories, rtree_idx):
    """
    Find pairs of trajectories that have overlapping spatial envelopes.
    """
    potential_pairs_list = []
    traj_ids = list(trajectories.keys())

    for i, traj_id1 in enumerate(traj_ids):
        traj1_points = trajectories[traj_id1]
        min_x, min_y = np.min(traj1_points, axis=0)
        max_x, max_y = np.max(traj1_points, axis=0)

        # Broad phase: find intersections in R-Tree
        hits = rtree_idx.intersection((min_x, min_y, max_x, max_y), objects=True)

        for hit in hits:
            traj_id2 = hit.object
            if traj_id1 < traj_id2:  # Avoid duplicate pairs and self-intersection
                potential_pairs_list.append((traj_id1, traj_id2))

    return potential_pairs_list


def compute_bbox_collisions(traj_pair, trajectories, times, vehicle_data, time_threshold=3.0):
    """
    Identify bounding box intersection events between a pair of trajectories.
    Uses a time threshold to determine if vehicles occupy the same space within a duration.
    """
    traj_id1, traj_id2 = traj_pair
    points1, points2 = trajectories[traj_id1], trajectories[traj_id2]
    times1, times2 = times[traj_id1], times[traj_id2]
    vdata1, vdata2 = vehicle_data[traj_id1], vehicle_data[traj_id2]

    results = []

    # Check for spatial overlap at each time step
    for i, t1 in enumerate(times1):
        # Find the point in the second trajectory closest in time
        time_diffs = np.abs(times2 - t1)
        min_diff = np.min(time_diffs)

        if min_diff <= time_threshold:
            j = np.argmin(time_diffs)

            # Positions
            x1, y1 = points1[i]
            x2, y2 = points2[j]

            # Bounding box corner calculation
            corners1 = get_bbox_corners(x1, y1, vdata1['hx'][i], vdata1['hy'][i],
                                        vdata1['length'], vdata1['width'])
            corners2 = get_bbox_corners(x2, y2, vdata2['hx'][j], vdata2['hy'][j],
                                        vdata2['length'], vdata2['width'])

            # Intersection check using SAT
            if check_bbox_collision(corners1, corners2):
                # Calculate the approximate collision center
                collision_x = (x1 + x2) / 2
                collision_y = (y1 + y2) / 2

                results.append({
                    'intersect_x': np.round(collision_x, 2),
                    'intersect_y': np.round(collision_y, 2),
                    'id1': traj_id1,
                    'id2': traj_id2,
                    'time1': times1[i],
                    'time2': times2[j],
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'angle1': vdata1['angle'][i],
                    'angle2': vdata2['angle'][j]
                })
    return results


def compute_pet(df, results_list, id_col='uid', type_col='type'):
    """
    Finalize results by mapping vehicle types and calculating PET values.
    """
    if len(results_list) == 0:
        return pd.DataFrame()
        
    result_df = pd.DataFrame(results_list)
    
    # Map ID to vehicle type
    id_to_type = df.set_index(id_col)[type_col].to_dict() if type_col in df.columns else {}
    if id_to_type:
        result_df['type1'] = result_df['id1'].map(id_to_type)
        result_df['type2'] = result_df['id2'].map(id_to_type)
        
    # Calculate PET (Post-Encroachment Time)
    result_df['PET'] = np.abs(result_df['time1'] - result_df['time2'])
    result_df['PET'] = result_df['PET'].astype(float).round(2)

    return result_df


def bbox_collisions_main(file_path, id_col='uid', time_col='time', x_col='longitude', y_col='latitude',
                         angle_col='angle', speed_col='speed', length_col='length', width_col='width',
                         type_col='type', drop_types=None, time_threshold=3.0):
    """
    Main orchestrator for bounding box intersection detection.
    """
    # Load data
    trajectories, times, vehicle_data, data = load_data(
        file_path, id_col, time_col, x_col, y_col, angle_col, speed_col, length_col, width_col, type_col, drop_types
    )

    # Build spatial index
    rtree_idx = build_rtree(trajectories)

    # Find candidate pairs
    pairs = potential_pairs(trajectories, rtree_idx)
    print(f"Found {len(pairs)} potential intersection pairs.")

    # Parallel computation of intersections
    with Pool(max(1, cpu_count() - 2)) as pool:
        results = list(tqdm(
            pool.starmap(
                compute_bbox_collisions,
                [(pair, trajectories, times, vehicle_data, time_threshold) for pair in pairs]
            ),
            total=len(pairs),
            desc="Processing trajectory pairs"
        ))

    # Flatten the result list
    final_results = [item for sublist in results for item in sublist]

    # Calculate PET and wrap up
    if final_results:
        result_df = compute_pet(data, final_results, id_col, type_col)
        print(f"Total found {len(result_df)} bounding box intersection events.")
        return result_df
    else:
        print("No bounding box intersection events found.")
        return pd.DataFrame()


if __name__ == "__main__":
    # Batch process all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_name = os.path.splitext(file_name)[0] + "_PET.csv"
            output_file_path = os.path.join(output_dir, output_file_name)

            collision_results = bbox_collisions_main(
                input_file_path,
                id_col='id',
                time_col='time',
                x_col='cx_m',
                y_col='cy_m',
                angle_col='yaw',    # Orientation angle
                speed_col='speed',
                length_col='w_m',    # Instantaneous vehicle length
                width_col='h_m',     # Instantaneous vehicle width
                type_col='type',
                time_threshold=5.0   # Time sync threshold in seconds
            )

            # Save results if not empty
            if not collision_results.empty:
                collision_results.to_csv(output_file_path, index=False)