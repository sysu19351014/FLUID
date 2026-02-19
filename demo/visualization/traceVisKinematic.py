# -*- coding: utf-8 -*-
"""
@Time    : 2026/2/5 15:51
@Author  : Terry CYY
@FileName: traceVisKinematic.py
@Software: PyCharm
@Function: Trajectory visualization script supporting Horizontal Bounding Box (HBB) 
           and Rotated Bounding Box (OBB) display. Shows ID, type, speed (km/h), 
           and direction. Features configurable trail length to avoid memory issues.
"""
import cv2
import pandas as pd
import numpy as np
import math
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Trajectory Visualizer for video data."""

    def __init__(self, config=None):
        """Initialize the visualizer with optional configurations."""
        self.config = config or {}

        # Default configurations
        self.default_config = {
            'resolution_scale': 0.5,  # Resolution scale (1.0 for original resolution)
            'box_thickness': 2,
            'text_scale': 0.6,
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'short_trail_length': 30,  # Short trail length (number of frames)
            'trail_thickness': 2,
            'show_speed': True,
            'speed_column': 'speed_smooth',  # Velocity column name
            'show_direction': True,
            'speed_unit': 'km/h',  # Speed display unit
        }

        # Update configuration
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        # Color configurations (BGR format)
        self.colors = {
            'pedestrian': (0, 180, 0),    # Green
            'moped': (180, 0, 0),         # Blue
            'car': (0, 165, 255),         # Orange
            'other': (0, 165, 255)        # Orange
        }

        # Text background colors (RGBA-like for semi-transparency)
        self.text_bg_colors = {
            'pedestrian': (50, 180, 50, 0.7),
            'moped': (180, 50, 50, 0.7),
            'car': (0, 140, 220, 0.7),
            'other': (0, 140, 220, 0.7)
        }

        # Trajectory storage
        self.full_trajectories = defaultdict(list)   # Full trajectory (unlimited length)
        self.short_trajectories = defaultdict(list)  # Short trail (limited length)

    def get_color(self, obj_type):
        """Get color based on object type."""
        obj_type_lower = obj_type.lower() if isinstance(obj_type, str) else 'other'

        if 'pedestrian' in obj_type_lower:
            return self.colors['pedestrian']
        elif 'moped' in obj_type_lower or 'motor' in obj_type_lower:
            return self.colors['moped']
        elif 'car' in obj_type_lower:
            return self.colors['car']
        else:
            return self.colors['other']

    def get_text_bg_color(self, obj_type):
        """Get text background color based on object type."""
        obj_type_lower = obj_type.lower() if isinstance(obj_type, str) else 'other'

        if 'pedestrian' in obj_type_lower:
            return self.text_bg_colors['pedestrian']
        elif 'moped' in obj_type_lower or 'motor' in obj_type_lower:
            return self.text_bg_colors['moped']
        elif 'car' in obj_type_lower:
            return self.text_bg_colors['car']
        else:
            return self.text_bg_colors['other']

    def scale_coordinates(self, cx, cy, w, h, original_size, target_size):
        """Scale coordinates and dimensions based on resolution change."""
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

        return (
            int(cx * scale_x),
            int(cy * scale_y),
            int(w * scale_x),
            int(h * scale_y)
        )

    def update_trajectories(self, obj_id, obj_type, cx, cy):
        """Update trajectory points based on object type."""
        obj_type_lower = obj_type.lower() if isinstance(obj_type, str) else 'other'

        # For VRUs (pedestrian/moped): Only update short trails to avoid visual clutter
        if 'pedestrian' in obj_type_lower or 'moped' in obj_type_lower:
            if obj_id not in self.short_trajectories:
                self.short_trajectories[obj_id] = []
            self.short_trajectories[obj_id].append((cx, cy))

            # Limit short trail length
            if len(self.short_trajectories[obj_id]) > self.config['short_trail_length']:
                self.short_trajectories[obj_id].pop(0)
        # For other types (e.g., cars): Update full trajectory
        else:
            self.full_trajectories[obj_id].append((cx, cy))

    def draw_full_trajectory(self, frame, obj_id):
        """Draw full trajectory lines for motor vehicles."""
        if obj_id not in self.full_trajectories or len(self.full_trajectories[obj_id]) < 2:
            return

        points = self.full_trajectories[obj_id]
        color = (0, 165, 255)  # Orange
        points_array = np.array(points, np.int32)

        # Draw trajectory line with anti-aliasing
        cv2.polylines(frame, [points_array], False, color,
                      self.config['trail_thickness'], cv2.LINE_AA)

        # Draw markers at key intervals
        step = max(1, len(points) // 20)
        for i in range(0, len(points), step):
            cv2.circle(frame, points[i], 2, color, -1)

        # Draw current position marker
        cv2.circle(frame, points[-1], 5, color, -1)
        cv2.circle(frame, points[-1], 3, (255, 255, 255), -1)

    def draw_short_trajectory(self, frame, obj_id):
        """Draw short fading trails for VRUs."""
        if obj_id not in self.short_trajectories or len(self.short_trajectories[obj_id]) < 2:
            return

        points = self.short_trajectories[obj_id]

        # Draw trail lines with alpha/thickness gradient
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(self.config['trail_thickness'] * alpha))
            cv2.line(frame, points[i - 1], points[i], (255, 255, 255),
                     thickness, cv2.LINE_AA)

        if points:
            cv2.circle(frame, points[-1], 4, (255, 255, 255), -1)

    def draw_semi_transparent_rect(self, frame, x1, y1, x2, y2, color_rgb, alpha):
        """Draw a semi-transparent rectangle for text backgrounds."""
        overlay = frame.copy()
        rgb_color = color_rgb[:3] if len(color_rgb) == 4 else color_rgb

        cv2.rectangle(overlay, (x1, y1), (x2, y2), rgb_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rgb_color, 1)

    def draw_obb(self, frame, cx, cy, w, h, angle, obj_id, obj_type, speed=None, direction=None):
        """
        Draw Rotated Bounding Box (OBB) and related kinematics info.
        angle: Rotation in radians.
        speed: Velocity in m/s.
        direction: Flow direction string.
        """
        box_color = self.get_color(obj_type)
        text_bg_color_info = self.get_text_bg_color(obj_type)

        # Update and draw trajectories based on type
        self.update_trajectories(obj_id, obj_type, cx, cy)
        obj_type_lower = obj_type.lower() if isinstance(obj_type, str) else 'other'
        if 'pedestrian' in obj_type_lower or 'moped' in obj_type_lower:
            self.draw_short_trajectory(frame, obj_id)
        else:
            self.draw_full_trajectory(frame, obj_id)

        # Convert to OpenCV RotatedRect format
        angle_deg = math.degrees(angle)
        rect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw OBB contour
        cv2.drawContours(frame, [box], 0, box_color, self.config['box_thickness'])

        # Process Direction info
        show_direction = False
        direction_text = ""
        if self.config['show_direction'] and direction is not None:
            if "Unknown" not in direction and direction.strip() != "":
                show_direction = True
                direction_text = direction

        # Process Speed info
        show_speed = False
        speed_text = ""
        if self.config['show_speed'] and speed is not None and not np.isnan(speed):
            show_speed = True
            if self.config['speed_unit'] == 'km/h':
                speed_display = speed * 3.6  # m/s to km/h
                speed_text = f"{speed_display:.1f}km/h"
            else:
                speed_text = f"{speed:.1f}m/s"

        id_type_text = f"{obj_id}:{obj_type}"
        font = self.config['font']
        text_scale = self.config['text_scale']
        thickness = 1

        # Text size calculations
        (text_w_id, text_h_id), _ = cv2.getTextSize(id_type_text, font, text_scale, thickness)
        text_w_speed, text_h_speed = (0, 0)
        if show_speed:
            (text_w_speed, text_h_speed), _ = cv2.getTextSize(speed_text, font, text_scale, thickness)
        text_w_dir, text_h_dir = (0, 0)
        if show_direction:
            (text_w_dir, text_h_dir), _ = cv2.getTextSize(direction_text, font, text_scale, thickness)

        max_text_w = max(text_w_id, text_w_speed, text_w_dir)

        # Top Label (ID and Type)
        bg_x1_top = int(cx - max_text_w / 2)
        bg_y1_top = int(cy - h / 2 - text_h_id - 10)
        bg_x2_top = int(cx + max_text_w / 2)
        bg_y2_top = bg_y1_top + text_h_id + 5

        # Boundary check for top label
        frame_h, frame_w = frame.shape[:2]
        if bg_y1_top < 0:
            bg_y1_top = int(cy + h / 2 + 5)
            bg_y2_top = bg_y1_top + text_h_id + 5

        if bg_x2_top > bg_x1_top + 5 and bg_y2_top > bg_y1_top + 5:
            self.draw_semi_transparent_rect(frame, bg_x1_top, bg_y1_top,
                                            bg_x2_top, bg_y2_top, text_bg_color_info,
                                            text_bg_color_info[3])
            cv2.putText(frame, id_type_text, (bg_x1_top, bg_y1_top + text_h_id + 3),
                        font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Bottom Label (Speed and/or Direction)
        bottom_texts = []
        if show_speed: bottom_texts.append(speed_text)
        if show_direction: bottom_texts.append(direction_text)

        if bottom_texts:
            bottom_text = " | ".join(bottom_texts)
            (text_w_bottom, text_h_bottom), _ = cv2.getTextSize(bottom_text, font, text_scale, thickness)

            bg_x1_bottom = int(cx - text_w_bottom / 2)
            bg_y1_bottom = int(cy + h / 2 + 5)
            bg_x2_bottom = int(cx + text_w_bottom / 2)
            bg_y2_bottom = bg_y1_bottom + text_h_bottom + 5

            if bg_y2_bottom > frame_h - 5:
                bg_y2_bottom = int(cy - h / 2 - 5)
                bg_y1_bottom = bg_y2_bottom - text_h_bottom - 5

            if bg_x2_bottom > bg_x1_bottom + 5 and bg_y2_bottom > bg_y1_bottom + 5:
                self.draw_semi_transparent_rect(frame, bg_x1_bottom, bg_y1_bottom,
                                                bg_x2_bottom, bg_y2_bottom, text_bg_color_info,
                                                text_bg_color_info[3])
                cv2.putText(frame, bottom_text, (bg_x1_bottom, bg_y1_bottom + text_h_bottom + 3),
                            font, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def process_video(self, csv_path, input_video_path, output_video_path=None):
        """Main processing loop: read CSV, read Video, and visualize."""
        self.full_trajectories.clear()
        self.short_trajectories.clear()

        logger.info(f"Reading CSV: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['frame', 'id', 'type', 'smooth_cx', 'smooth_cy', 'w', 'h', 'smooth_r']
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                logger.error(f"CSV missing columns: {missing_cols}")
                return False

            df['frame'] = df['frame'].astype(int)
            grouped = df.groupby('frame')

            speed_col = self.config['speed_column']
            has_speed = speed_col in df.columns
            has_direction = 'overall_direction' in df.columns

            logger.info("Object type counts:")
            for type_name, count in df['type'].value_counts().items():
                logger.info(f"  {type_name}: {count}")

            logger.info(f"Loaded {len(df)} rows, {df['id'].nunique()} objects.")
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return False

        logger.info(f"Opening Video: {input_video_path}")
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            logger.error("Could not open video file.")
            return False

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_width = int(original_width * self.config['resolution_scale'])
        target_height = int(original_height * self.config['resolution_scale'])

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (target_width, target_height))
            logger.info(f"Exporting to: {output_video_path}")

        pbar = tqdm(total=total_frames, unit='frames', desc='Visualizing')
        frame_num = 0
        success = True

        try:
            while success:
                success, frame = cap.read()
                if not success: break
                frame_num += 1

                if target_width != original_width or target_height != original_height:
                    frame = cv2.resize(frame, (target_width, target_height))

                if frame_num in grouped.groups:
                    detections = grouped.get_group(frame_num)
                    for _, row in detections.iterrows():
                        cx, cy, w, h = self.scale_coordinates(
                            row['smooth_cx'], row['smooth_cy'], row['w'], row['h'],
                            (original_width, original_height), (target_width, target_height)
                        )
                        speed = row.get(self.config['speed_column']) if has_speed else None
                        direction = row.get('overall_direction') if has_direction else None

                        self.draw_obb(frame, cx, cy, w, h, row['smooth_r'], int(row['id']), str(row['type']), speed, direction)

                cv2.putText(frame, f'Frame: {frame_num}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if writer:
                    writer.write(frame)
                else:
                    cv2.imshow('Trajectory Visualization', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                pbar.update(1)

        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            pbar.close()
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            logger.info("Processing complete.")
            return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kinematic Trajectory Visualization Tool')
    parser.add_argument('--csv', type=str, required=True, help='Path to trajectory CSV file')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')
    parser.add_argument('--scale', type=float, default=0.5, help='Resolution scaling factor (default: 0.5)')
    parser.add_argument('--trail-length', type=int, default=30, help='Short trail length in frames (default: 30)')
    parser.add_argument('--no-speed', action='store_true', help='Disable speed display')
    parser.add_argument('--speed-column', type=str, default='speed', help='Speed column name (default: speed)')
    parser.add_argument('--no-direction', action='store_true', help='Disable direction display')
    parser.add_argument('--speed-unit', choices=['km/h', 'm/s'], default='km/h', help='Speed unit (default: km/h)')
    return parser.parse_args()


def main():
    """Main execution entry."""
    args = parse_args()
    config = {
        'resolution_scale': args.scale,
        'short_trail_length': args.trail_length,
        'show_speed': not args.no_speed,
        'speed_column': args.speed_column,
        'show_direction': not args.no_direction,
        'speed_unit': args.speed_unit
    }
    visualizer = TrajectoryVisualizer(config)
    visualizer.process_video(csv_path=args.csv, input_video_path=args.video, output_video_path=args.output)


if __name__ == "__main__":
    main()