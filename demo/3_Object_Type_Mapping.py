# -*- coding: utf-8 -*-
"""
@Time    : 2025/7/4 12:19
@Author  : Terry CYY
@FileName: 3_Object_Type_Mapping.py
@Software: PyCharm
@Function: Map the 'type' to different categories based on the string contained in the filename 
            (which indicates that the file was inferred using model weights trained on different datasets).
"""
import os
import numpy as np
import pandas as pd

# Define mapping rules
visdrone_type = {
    0: 'pedestrian',
    1: 'bicycle',
    2: 'car',
    3: 'van',
    4: 'truck',
    5: 'tricycle',
    6: 'bus',
    7: 'motor'
}

dronevehicle_type = {
    0: 'bus',
    1: 'car',
    2: 'freight car',
    3: 'truck',
    4: 'van',
    5: 'bicycle',
    6: 'tricycle'
}

songdo_type = {
    0: 'car',
    1: 'bus',
    2: 'truck',
    3: 'motorcycle'
}

# When training with all target types in the CoDrone dataset
codrone_type = {  
    0: 'car',
    1: 'truck',
    2: 'traffic-sign',
    3: 'people',
    4: 'motor',
    5: 'bicycle',
    6: 'traffic-light',
    7: 'tricycle',
    8: 'bridge',
    9: 'bus',
    10: 'boat',
    11: 'ship'
}

codroneLess_type = {
    0: 'car',
    1: 'truck',
    2: 'people',
    3: 'motorcycle',
    4: 'bicycle',
    5: 'tricycle',
    6: 'bus',
}

def map_type(file):
    """
    Batch map target types
    """
    # Determine the mapping rule based on the filename
    if 'visdrone' in file:
        mapping = visdrone_type
    elif 'dronevehicle' in file:
        mapping = dronevehicle_type
    elif 'songdo' in file:
        mapping = songdo_type
    elif 'codroneLess' in file:
        mapping = codroneLess_type
    else:
        print(f"Unrecognized file type: {file}")
        return file  # Return the original data without any modification
    # Read data
    data = pd.read_csv(os.path.join(folder, file))
    # Convert the 'type' column to int
    data['type'] = pd.to_numeric(data['type'], errors='coerce').fillna(-1).astype(int)
    # Map the 'type' column
    if 'type' in data.columns:
        data['type'] = data['type'].map(mapping).fillna('unknown')
    else:
        print(f"'type' column not found in file {file}")

    return data

# Target path
folder = 'output/test'


if __name__ == '__main__':
        # Get all CSV files
        csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
        print(csv_files)
        for csv_file in csv_files:
            mapped_data = map_type(csv_file)
            if isinstance(mapped_data, pd.DataFrame):
                # Save the mapped data
                output_file = os.path.join(folder, f"{csv_file}")
                mapped_data.to_csv(output_file, index=False)
                print(f"Mapped data has been saved to: {output_file}")
            else:
                print(f"Skip processing file {csv_file}")