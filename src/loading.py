"""File containing logic of loading raw data into usable numpy arrays.

Usage Example:

    x, y = load_data("xor.csv")
"""
import os
import numpy as np

def load_data(data: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the datasets from raw .csv files.

    Args:
        data: The name of the .csv file to load.

    Returns:
        tuple[np.ndarray, np.ndarray]: Input and output of the data.
    """
    # folder of this file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # go back one step (into the project root), then into the data folder
    root_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # make sure data exists
    existing_files = set(os.listdir(data_dir))
    if data not in existing_files:
        raise FileNotFoundError(f"Data ({data}) does not exist in {data_dir}.")

    data_file = os.path.join(data_dir, data)
    try:
        data = np.genfromtxt(data_file, delimiter=",", names=True, dtype=float)
    except Exception as e:
        print(f"Error loading file {data}:", e)

    # get column names
    names = data.dtype.names

    # use all fields as input except for y
    feature_fields = [n for n in names if n != 'y']
    x = np.column_stack([data[n] for n in feature_fields])

    # use y field as label
    y = data['y'] if 'y' in names else data[names[-1]]
    y = y.reshape(-1, 1)

    # shape (batches, 2) % (batches, 1)
    return (x, y)
