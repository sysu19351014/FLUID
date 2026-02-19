# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/15 4:15
@Author  : Terry_CYY
@File    : 2_droneTwoDimSSM.py
@IDE     : PyCharm
@Function: Calculate 2D Surrogate Safety Measures (mainly TTC, DRAC, and MTTC) 
           for drone trajectories with optimized performance.
"""
import numpy as np
import pandas as pd
import warnings
from scipy.spatial import KDTree
from pyproj import CRS, Transformer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import glob
import os
import argparse

warnings.filterwarnings("ignore")

# 1. Default Configuration
"""-------------- FLUID Dataset Mapping ------------------"""
COLUMN_MAP = {
    'id': 'id',
    'x': 'cx_m',      # Use UTM for easier visualization
    'y': 'cy_m',
    'angle': 'yaw',
    'speed': ['vx', 'vy'],  # Supports multiple columns (or single column)
    'length': 'w_m',        # Instantaneous vehicle length
    'width': 'h_m',         # Instantaneous vehicle width
    'frame': 'frame',
    'time': 'time',
    'accel': ['ax', 'ay'],
}

# Default Folders
DEFAULT_INPUT_DIR = r"result"
DEFAULT_OUTPUT_DIR = r"conflict"

# Argument Parsing
parser = argparse.ArgumentParser(description="Calculate 2D SSMs for drone trajectory data.")
parser.add_argument('--input_dir', type=str, help='Input directory containing CSV files')
parser.add_argument('--output_dir', type=str, help='Output directory for processed CSV files')
args = parser.parse_args()

input_dir = args.input_dir if args.input_dir else DEFAULT_INPUT_DIR
output_dir = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

# Batch processing setup
input_files = glob.glob(os.path.join(input_dir, '*.csv'))


class GeometryUtils:
    def __init__(self, tol=1e-5):
        """
        Initialize geometry utilities with a tolerance threshold.
        """
        self.tol = tol

    @staticmethod
    def line(point0, point1):
        """
        Get the line equation (ax + by + c = 0) from two points.
        """
        x0, y0 = point0
        x1, y1 = point1
        a = y0 - y1
        b = x1 - x0
        c = x0 * y1 - x1 * y0
        return a, b, c

    @staticmethod
    def intersect(line0, line1):
        """
        Find the intersection coordinates of two lines.
        """
        a0, b0, c0 = line0
        a1, b1, c1 = line1
        D = a0 * b1 - a1 * b0
        # Avoid division by zero by setting to NaN
        D_is_zero = (D == 0)
        x = (b0 * c1 - b1 * c0) / np.where(D_is_zero, np.nan, D)
        y = (a1 * c0 - a0 * c1) / np.where(D_is_zero, np.nan, D)
        return np.array([x, y])

    def is_on(self, line_start, line_end, point):
        """
        Check if a point lies on a specific line segment within a tolerance.
        """
        crossproduct = (point[1] - line_start[1]) * (line_end[0] - line_start[0]) - (point[0] - line_start[0]) * (
                line_end[1] - line_start[1])
        dotproduct = (point[0] - line_start[0]) * (line_end[0] - line_start[0]) + (point[1] - line_start[1]) * (
                line_end[1] - line_start[1])
        squaredlength = (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2
        return (np.absolute(crossproduct) <= self.tol) & (dotproduct >= 0) & (dotproduct <= squaredlength)

    @staticmethod
    def dist_p2l(point, line_start, line_end):
        """
        Calculate the distance from a point to a line.
        """
        num = np.absolute((line_end[0] - line_start[0]) * (line_start[1] - point[1]) - 
                          (line_start[0] - point[0]) * (line_end[1] - line_start[1]))
        den = np.sqrt((line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2)
        return num / den

    @staticmethod
    def get_points(samples):
        """
        Retrieve the coordinates of the four corners of the bounding boxes for vehicles i and j.
        """
        samples['id_i'] = samples['id_i'].astype(int)
        samples['id_j'] = samples['id_j'].astype(int)

        # Vehicle i
        heading_i = samples[['hx_i', 'hy_i']].values
        perp_heading_i = np.array([-heading_i[:, 1], heading_i[:, 0]]).T
        heading_scale_i = np.tile(np.sqrt(heading_i[:, 0] ** 2 + heading_i[:, 1] ** 2), (2, 1)).T
        length_i = np.tile(samples.length_i.values, (2, 1)).T
        width_i = np.tile(samples.width_i.values, (2, 1)).T

        point_up = samples[['x_i', 'y_i']].values + heading_i / heading_scale_i * length_i / 2
        point_down = samples[['x_i', 'y_i']].values - heading_i / heading_scale_i * length_i / 2
        point_i1 = (point_up + perp_heading_i / heading_scale_i * width_i / 2).T
        point_i2 = (point_up - perp_heading_i / heading_scale_i * width_i / 2).T
        point_i3 = (point_down + perp_heading_i / heading_scale_i * width_i / 2).T
        point_i4 = (point_down - perp_heading_i / heading_scale_i * width_i / 2).T

        # Vehicle j
        heading_j = samples[['hx_j', 'hy_j']].values
        perp_heading_j = np.array([-heading_j[:, 1], heading_j[:, 0]]).T
        heading_scale_j = np.tile(np.sqrt(heading_j[:, 0] ** 2 + heading_j[:, 1] ** 2), (2, 1)).T
        length_j = np.tile(samples.length_j.values, (2, 1)).T
        width_j = np.tile(samples.width_j.values, (2, 1)).T

        point_up = samples[['x_j', 'y_j']].values + heading_j / heading_scale_j * length_j / 2
        point_down = samples[['x_j', 'y_j']].values - heading_j / heading_scale_j * length_j / 2
        point_j1 = (point_up + perp_heading_j / heading_scale_j * width_j / 2).T
        point_j2 = (point_up - perp_heading_j / heading_scale_j * width_j / 2).T
        point_j3 = (point_down + perp_heading_j / heading_scale_j * width_j / 2).T
        point_j4 = (point_down - perp_heading_j / heading_scale_j * width_j / 2).T

        return point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4


def calculate_vector(df):
    """
    Calculate velocity components and heading vectors.
    """
    # Check if 'angle' is in degrees (if absolute max exceeds 2*PI)
    if df['angle'].abs().max() > 6.3:
        df['angle'] = df['angle'].round(2)
        df['angle'] = np.deg2rad(df['angle'])
    
    # Unit length (d) for the direction vector
    d = 1.0  
    # Calculate hx, hy (Heading direction vectors)
    df['hx'] = d * np.cos(df['angle'])
    df['hy'] = d * np.sin(df['angle'])
    # Decompose speed into vx, vy
    df['vx'] = df['speed'] * np.cos(df['angle'])
    df['vy'] = df['speed'] * np.sin(df['angle'])


def coord_transform(df, transformer):
    """
    Transform Lat/Lon to projected coordinates (UTM). Uses existing x/y if available.
    """
    if 'x' in df.columns and 'y' in df.columns:
        return
    else:
        df['x'], df['y'] = transformer.transform(df['longitude'], df['latitude'])


def CurrentD(samples, toreturn='dataframe'):
    """
    Calculate the minimum distance between bounding boxes of vehicle i and j.
    Returns 0 if boxes overlap.
    """
    geo = GeometryUtils(tol=1e-2)
    if toreturn not in ['dataframe', 'values']:
        warnings.warn("Unknown return type. Please specify 'dataframe' or 'values'.")
        return None
    
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = geo.get_points(samples)

    dist_mat = []
    count_i = 0
    # Iterate through edges of vehicle i
    for point_i_start, point_i_end in zip([point_i1, point_i4, point_i3, point_i2],
                                          [point_i2, point_i3, point_i1, point_i4]):
        count_j = 0
        # Iterate through edges of vehicle j
        for point_j_start, point_j_end in zip([point_j1, point_j4, point_j3, point_j2],
                                              [point_j2, point_j3, point_j1, point_j4]):
            if count_i < 2 and count_j < 2:
                # Vertex-to-vertex distances
                for pi in [point_i_start, point_i_end]:
                    for pj in [point_j_start, point_j_end]:
                        dist_mat.append(np.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2))

            # Point-to-line distances
            perp_vec = np.array([-(point_j_start - point_j_end)[1], (point_j_start - point_j_end)[0]])
            ist = geo.intersect(geo.line(point_i_start, point_i_start + perp_vec),
                                geo.line(point_j_start, point_j_end))
            ist[:, ~geo.is_on(point_j_start, point_j_end, ist)] = np.nan
            dist_mat.append(np.sqrt((ist[0] - point_i_start[0]) ** 2 + (ist[1] - point_i_start[1]) ** 2))

            # Identify overlapping bounding boxes
            ist_overlap = geo.intersect(geo.line(point_i_start, point_i_end),
                                        geo.line(point_j_start, point_j_end))
            dist = np.ones(len(samples)) * np.nan
            dist[geo.is_on(point_i_start, point_i_end, ist_overlap) &
                 geo.is_on(point_j_start, point_j_end, ist_overlap)] = 0
            dist[np.isnan(ist_overlap[0]) & (geo.is_on(point_i_start, point_i_end, point_j_start) |
                                             geo.is_on(point_i_start, point_i_end, point_j_end))] = 0
            dist_mat.append(dist)
            count_j += 1
        count_i += 1

    cdist = np.nanmin(np.array(dist_mat), axis=0)

    if toreturn == 'dataframe':
        samples['CurrentD'] = cdist
        return samples
    return cdist


def DTC_ij(samples):
    """
    Calculate the Distance to Collision (DTC) between two vehicles 
    and determine if they are receding (moving away) from each other.
    """
    geo = GeometryUtils(tol=1e-2)
    point_i1, point_i2, point_i3, point_i4, point_j1, point_j2, point_j3, point_j4 = geo.get_points(samples)
    relative_v = (samples[['vx_i', 'vy_i']].values - samples[['vx_j', 'vy_j']].values).T

    dist_mat = []
    leaving_mat = []
    # For each corner of vehicle i
    for point_line_start in [point_i1, point_i2, point_i3, point_i4]:
        # For each edge of vehicle j
        for edge_start, edge_end in zip([point_j1, point_j3, point_j1, point_j2],
                                        [point_j2, point_j4, point_j3, point_j4]):
            point_line_end = point_line_start + relative_v
            # Intersection of edge j and a line extended from corner i along relative velocity
            ist = geo.intersect(geo.line(point_line_start, point_line_end), geo.line(edge_start, edge_end))
            ist[:, ~geo.is_on(edge_start, edge_end, ist)] = np.nan
            
            # Distance from point i to intersection
            dist_ist = np.sqrt((ist[0] - point_line_start[0]) ** 2 + (ist[1] - point_line_start[1]) ** 2)
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            
            # Determine if receding (scalar product of relative velocity and vector to intersection)
            dot_prod = relative_v[0] * (ist[0] - point_line_start[0]) + relative_v[1] * (ist[1] - point_line_start[1])
            # 1: Receding/Moving away, 20: Closing/Not receding
            leaving = np.where(dot_prod >= 0, 20, 1)
            leaving_mat.append(leaving)

    dtc = np.array(dist_mat).min(axis=0)
    leaving = np.nansum(np.array(leaving_mat), axis=0)
    return dtc, leaving


def TTC(samples, toreturn='dataframe'):
    """
    Calculate Time to Collision (TTC).
    """
    delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
    dtc_ij, leaving_ij = DTC_ij(samples)
    ttc_ij = dtc_ij / delta_v
    ttc_ij[leaving_ij < 20] = np.inf  # Vehicles will not collide if they are moving away
    ttc_ij[(leaving_ij > 20) & (leaving_ij % 20 != 0)] = -1  # -1 indicates overlapping bounding boxes

    # Repeat calculation by swapping vehicle roles (ji)
    keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    keys.extend(values); values.extend(keys)
    rename_dict = {keys[i]: values[i] for i in range(len(keys))}
    dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
    ttc_ji = dtc_ji / delta_v
    ttc_ji[leaving_ji < 20] = np.inf
    ttc_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

    res = np.minimum(ttc_ij, ttc_ji)
    if toreturn == 'dataframe':
        samples['TTC'] = res
        return samples
    return res


def DRAC(samples, toreturn='dataframe'):
    """
    Calculate Deceleration Rate to Avoid Collision (DRAC).
    """
    delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
    dtc_ij, leaving_ij = DTC_ij(samples)
    drac_ij = delta_v ** 2 / dtc_ij / 2
    drac_ij[leaving_ij < 20] = 0.
    drac_ij[(leaving_ij > 20) & (leaving_ij % 20 != 0)] = -1

    keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    keys.extend(values); values.extend(keys)
    rename_dict = {keys[i]: values[i] for i in range(len(keys))}
    dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
    drac_ji = delta_v ** 2 / dtc_ji / 2
    drac_ji[leaving_ji < 20] = 0.
    drac_ji[(leaving_ji > 20) & (leaving_ji % 20 != 0)] = -1

    res = np.maximum(drac_ij, drac_ji)
    if toreturn == 'dataframe':
        samples['DRAC'] = res
        return samples
    return float(res.iloc[0])


def MTTC(samples, toreturn='dataframe'):
    """
    Calculate Modified Time to Collision (MTTC) which incorporates acceleration.
    """
    if 'acc_i' not in samples.columns:
        warnings.warn('Acceleration not provided for current objects!')
        return None
    
    delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
    dtc_ij, leaving_ij = DTC_ij(samples)
    ttc_ij = dtc_ij / delta_v
    
    # Redo for swapped roles (ji)
    keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    keys.extend(values); values.extend(keys)
    rename_dict = {keys[i]: values[i] for i in range(len(keys))}
    dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
    ttc_ji = dtc_ji / delta_v

    ttc = np.minimum(ttc_ij, ttc_ji)
    dtc = np.minimum(dtc_ij, dtc_ji)

    if 'acc_j' in samples.columns:
        delta_a = samples['acc_i'].values - samples['acc_j'].values
    else:
        # Assume other vehicle maintains constant velocity
        delta_a = samples['acc_i'].values

    # Determine relative velocity sign based on closing/receding status
    is_closing = ((leaving_ij >= 20) | (leaving_ji >= 20)).astype(int)
    delta_v_signed = delta_v * np.sign(is_closing - 0.5)

    squared_term = delta_v_signed ** 2 + 2 * delta_a * dtc
    squared_term_valid = np.where(squared_term >= 0, np.sqrt(squared_term), np.nan)
    
    mttc_plus = (-delta_v_signed + squared_term_valid) / delta_a
    mttc_minus = (-delta_v_signed - squared_term_valid) / delta_a
    
    mttc = mttc_minus.copy()
    mttc[(mttc_minus <= 0) & (mttc_plus > 0)] = mttc_plus[(mttc_minus <= 0) & (mttc_plus > 0)]
    mttc[(mttc_minus <= 0) & (mttc_plus <= 0)] = np.inf
    mttc[np.isnan(mttc)] = np.inf
    mttc[abs(delta_a) < 1e-6] = ttc[abs(delta_a) < 1e-6]
    mttc[((leaving_ij > 20) & (leaving_ij % 20 != 0)) | ((leaving_ji > 20) & (leaving_ji % 20 != 0))] = -1

    if toreturn == 'dataframe':
        samples['MTTC'] = mttc
        return samples
    return mttc


def TTC_DRAC_MTTC(samples, toreturn='dataframe'):
    """
    Combined calculation for TTC, DRAC, and MTTC.
    """
    if 'acc_i' not in samples.columns:
        warnings.warn('Acceleration not provided!')
        return None

    delta_v = np.sqrt((samples['vx_i'] - samples['vx_j']) ** 2 + (samples['vy_i'] - samples['vy_j']) ** 2)
    dtc_ij, leaving_ij = DTC_ij(samples)
    ttc_ij = dtc_ij / delta_v
    ttc_ij[leaving_ij < 20] = np.inf
    ttc_ij[(leaving_ij > 20) & (leaving_ij % 20 != 0)] = -1
    
    drac_ij = delta_v ** 2 / dtc_ij / 2
    drac_ij[leaving_ij < 20] = 0.
    drac_ij[(leaving_ij > 20) & (leaving_ij % 20 != 0)] = -1

    keys = [var + '_i' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    values = [var + '_j' for var in ['x', 'y', 'vx', 'vy', 'hx', 'hy', 'length', 'width']]
    keys.extend(values); values.extend(keys)
    rename_dict = {keys[i]: values[i] for i in range(len(keys))}
    dtc_ji, leaving_ji = DTC_ij(samples.rename(columns=rename_dict))
    
    ttc_ji = dtc_ji / delta_v
    ttc_ji[leaving_ji < 20] = np.inf
    drac_ji = delta_v ** 2 / dtc_ji / 2
    drac_ji[leaving_ji < 20] = 0.

    dtc = np.minimum(dtc_ij, dtc_ji)
    ttc = np.minimum(ttc_ij, ttc_ji)
    drac = np.maximum(drac_ij, drac_ji)

    delta_a = samples['acc_i'].values - samples.get('acc_j', 0.0)
    is_closing = ((leaving_ij >= 20) | (leaving_ji >= 20)).astype(int)
    delta_v_signed = delta_v * np.sign(is_closing - 0.5)

    sq_term = delta_v_signed ** 2 + 2 * delta_a * dtc
    sq_term_valid = np.where(sq_term >= 0, np.sqrt(sq_term), np.nan)
    mttc_plus = (-delta_v_signed + sq_term_valid) / delta_a
    mttc_minus = (-delta_v_signed - sq_term_valid) / delta_a
    
    mttc = mttc_minus.copy()
    mttc[(mttc_minus <= 0) & (mttc_plus > 0)] = mttc_plus[(mttc_minus <= 0) & (mttc_plus > 0)]
    mttc[(mttc_minus <= 0) & (mttc_plus <= 0)] = np.inf
    mttc[np.isnan(mttc)] = np.inf
    mttc[abs(delta_a) < 1e-6] = ttc[abs(delta_a) < 1e-6]

    if toreturn == 'dataframe':
        samples['TTC'], samples['DRAC'], samples['MTTC'] = ttc, drac, mttc
        return samples
    return ttc, drac, mttc


def get_utm_transformer(df):
    """
    Determine the UTM projection based on the first row's coordinates.
    """
    lon, lat = df.iloc[0]['longitude'], df.iloc[0]['latitude']
    wgs84 = CRS("EPSG:4326")
    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    utm_crs = CRS.from_epsg(epsg_code)
    transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    return wgs84, utm_crs, transformer


def predict_ttc_collision_point(samples, t_pred):
    """Predict collision point based on linear motion (TTC)."""
    x1_pred = samples['x_i'].values + samples['vx_i'].values * t_pred
    y1_pred = samples['y_i'].values + samples['vy_i'].values * t_pred
    x2_pred = samples['x_j'].values + samples['vx_j'].values * t_pred
    y2_pred = samples['y_j'].values + samples['vy_j'].values * t_pred
    return x1_pred, y1_pred, x2_pred, y2_pred, (x1_pred + x2_pred)/2, (y1_pred + y2_pred)/2


def predict_mttc_collision_point(samples, t_pred):
    """Predict collision point based on accelerated motion (MTTC)."""
    acc_i = samples.get('acc_i', 0.0)
    acc_j = samples.get('acc_j', 0.0)
    x1_pred = samples['x_i'].values + samples['vx_i'].values * t_pred + 0.5 * acc_i * t_pred**2
    y1_pred = samples['y_i'].values + samples['vy_i'].values * t_pred + 0.5 * acc_i * t_pred**2
    x2_pred = samples['x_j'].values + samples['vx_j'].values * t_pred + 0.5 * acc_j * t_pred**2
    y2_pred = samples['y_j'].values + samples['vy_j'].values * t_pred + 0.5 * acc_j * t_pred**2
    return x1_pred, y1_pred, x2_pred, y2_pred, (x1_pred + x2_pred)/2, (y1_pred + y2_pred)/2


def process_frame(frame, frame_data):
    """
    Frame-by-frame 2D-SSM calculation using KDTree for spatial indexing.
    """
    if len(frame_data) < 2: return []
    if 'id' in frame_data.columns: frame_data['id'] = frame_data['id'].astype(int)

    positions = frame_data[['x', 'y']].values
    tree = KDTree(positions)
    # Search for pairs within 50 meters Euclidean distance
    pairs = list(tree.query_pairs(r=50))
    if not pairs: return []

    samples_list = []
    for (i, j) in pairs:
        c1, c2 = frame_data.iloc[i], frame_data.iloc[j]
        samples_list.append({
            'id_i': c1['id'], 'x_i': c1['x'], 'y_i': c1['y'], 'hx_i': c1['hx'], 'hy_i': c1['hy'],
            'vx_i': c1['vx'], 'vy_i': c1['vy'], 'length_i': c1['length'], 'width_i': c1['width'], 'acc_i': c1['accel'],
            'id_j': c2['id'], 'x_j': c2['x'], 'y_j': c2['y'], 'hx_j': c2['hx'], 'hy_j': c2['hy'],
            'vx_j': c2['vx'], 'vy_j': c2['vy'], 'length_j': c2['length'], 'width_j': c2['width'], 'acc_j': c2['accel']
        })

    samples_df = pd.DataFrame(samples_list)
    results_batch = TTC_DRAC_MTTC(samples_df, toreturn='dataframe')[['TTC', 'DRAC', 'MTTC']]
    ttc_v, drac_v, mttc_v = results_batch['TTC'].values, results_batch['DRAC'].values, results_batch['MTTC'].values

    _, _, _, _, x_mid_ttc, y_mid_ttc = predict_ttc_collision_point(samples_df, ttc_v)
    _, _, _, _, x_mid_mttc, y_mid_mttc = predict_mttc_collision_point(samples_df, mttc_v)

    results = []
    for idx, ((i, j), ttc, drac, mttc) in enumerate(zip(pairs, ttc_v, drac_v, mttc_v)):
        c1, c2 = frame_data.iloc[i], frame_data.iloc[j]
        
        ttc_x = round(x_mid_ttc[idx], 2) if np.isfinite(ttc) and ttc != -1 else np.nan
        ttc_y = round(y_mid_ttc[idx], 2) if np.isfinite(ttc) and ttc != -1 else np.nan
        mttc_x = round(x_mid_mttc[idx], 2) if np.isfinite(mttc) and mttc != -1 else np.nan
        mttc_y = round(y_mid_mttc[idx], 2) if np.isfinite(mttc) and mttc != -1 else np.nan

        results.append({
            'Frame': frame, 'Time': c1['time'], 'Car 1 ID': c1['id'], 'Car 2 ID': c2['id'],
            'Car 1 X': c1['x'], 'Car 1 Y': c1['y'], 'Car 2 X': c2['x'], 'Car 2 Y': c2['y'],
            'Car 1 Type': c1['type'], 'Car 2 Type': c2['type'], 'Car 1 Angle': c1['angle'], 'Car 2 Angle': c2['angle'],
            'DRAC': drac, 'TTC': ttc, 'MTTC': mttc,
            'TTC_Collision_X': ttc_x, 'TTC_Collision_Y': ttc_y,
            'MTTC_Collision_X': mttc_x, 'MTTC_Collision_Y': mttc_y
        })
    return results


def round_numeric(series):
    """Round finite, non-zero values to 2 decimal places."""
    return series.apply(lambda x: round(x, 2) if (x != 0 and np.isfinite(x)) else x)


if __name__ == '__main__':
    for input_path in tqdm(input_files, desc='Batch processing CSV files'):
        df = pd.read_csv(input_path)
        
        # Filter motor vehicles only
        df = df[~df['type'].isin(['moped', 'pedestrian'])]

        # Map columns based on configuration
        rename_dict = {v: k for k, v in COLUMN_MAP.items() if isinstance(v, str)}
        df = df.rename(columns=rename_dict)
        if 'id' in df.columns: df['id'] = df['id'].astype(int)

        # Handle multi-column speed/accel
        if isinstance(COLUMN_MAP['speed'], list):
            vx, vy = COLUMN_MAP['speed']
            if vx in df.columns and vy in df.columns:
                df['speed'] = np.sqrt(df[vx]**2 + df[vy]**2)

        if isinstance(COLUMN_MAP['accel'], list):
            ax_col, ay_col = COLUMN_MAP['accel']
            if ax_col in df.columns and ay_col in df.columns:
                df['accel'] = np.sqrt(df[ax_col]**2 + df[ay_col]**2)

        if 'longitude' in df.columns and 'latitude' in df.columns:
            _, _, transformer = get_utm_transformer(df)
            coord_transform(df, transformer)

        calculate_vector(df)

        # Multiprocessing across frames
        frame_groups = df.groupby('frame')
        with Pool(max(1, cpu_count() - 2)) as pool:
            results = list(tqdm(pool.starmap(process_frame, [(f, data) for f, data in frame_groups]),
                                total=len(frame_groups), leave=False))
        
        results_flat = [item for sublist in results for item in sublist]
        if not results_flat: continue
        
        results_df = pd.DataFrame(results_flat)
        results_df['Time'] = results_df['Time'].astype(float)
        for col in ['Car 1 ID', 'Car 2 ID']: results_df[col] = results_df[col].astype(int)
        for col in ['Car 1 X', 'Car 1 Y', 'Car 2 X', 'Car 2 Y']: results_df[col] = results_df[col].round(2)
        for col in ['DRAC', 'TTC', 'MTTC']: results_df[col] = round_numeric(results_df[col])

        # Filter out infinite interactions
        results_df = results_df[(results_df['TTC'] != float('inf')) & (results_df['MTTC'] != float('inf'))]

        # Output file
        base_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, base_name.replace('.csv', '_2D-SSM.csv'))
        results_df.to_csv(output_path, index=False)

    print("SSM calculation complete.")