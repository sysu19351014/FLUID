# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/1 14:02
@Author  : Terry CYY
@FileName: 1_Structured_Recognition_Results.py
@Software: PyCharm
@Function: For the original trajectory file output by object detection and tracking, first define the header, then read all txt files in the current directory in csv format, add the header and save it as a csv file
"""
import pandas as pd
import os


def add_header_to_hbb(header, directory='.'):
    """
    Add header to HBB results
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Get all txt files with hbb in their filenames
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt') and 'hbb' in f]

    if not txt_files:
        print("No hbb txt files found.")
        return

    i = 0  # Total number of files

    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)

        # Read txt file
        df = pd.read_csv(file_path, sep=',', header=None)

        # Add header
        df.columns = header

        # Calculate center point coordinates
        df['cx'] = df['ltx'] + df['w'] / 2
        df['cy'] = df['lty'] + df['h'] / 2

        # Keep only the first two decimal places
        df['cx'] = df['cx'].round(2)
        df['cy'] = df['cy'].round(2)

        # Save as csv format
        csv_file_path = os.path.splitext(file_path)[0] + '.csv'
        df.to_csv(csv_file_path, index=False)

        print(f"Converted {txt_file} to {csv_file_path}, added header, and added center point coordinates.")
        i += 1

    print(f'Processed {i} files in total!')

def add_header_to_obb(header, directory='.'):
    """
    Add header to OBB results
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Get all txt files with obb in their filenames
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt') and 'obb' in f]

    if not txt_files:
        print("No obb txt files found.")
        return

    i = 0  # Total number of files

    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)

        # Read txt file
        df = pd.read_csv(file_path, sep=',', header=None)

        # Add header
        df.columns = header

        # Save as csv format
        csv_file_path = os.path.splitext(file_path)[0] + '.csv'
        df.to_csv(csv_file_path, index=False)
        print(f"Converted {txt_file} to {csv_file_path} and added header.")
        i += 1

    print(f'Processed {i} files in total!')

if __name__ == "__main__":
    # The definition of the header depends on the output file
    # Process HBB (Horizontal Bounding Box) and OBB (Oriented Bounding Box) results separately
    add_header_to_hbb(['frame', 'id', 'ltx', 'lty', 'w', 'h', 'confidence', 'type'], directory='input')
    add_header_to_obb(['frame', 'id', 'cx', 'cy', 'w', 'h', 'r', 'confidence', 'type'], directory='input')