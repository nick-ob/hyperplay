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

    # try to load data
    data_file = os.path.join(data_dir, data)
    try:
        raw = np.genfromtxt(data_file, delimiter=",", names=True, dtype=float)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {data_file}") from e

    if raw is None:
        raise ValueError(f"Empty dataset: {data_file}")

    # verify correct dataset format
    raw = np.atleast_1d(raw)
    if raw.dtype.names is None:
        raise ValueError(f"Dataset must include a header row: {data_file}")

    names = list(raw.dtype.names)
    if not names:
        raise ValueError(f"Dataset has no columns: {data_file}")

    y_count = sum(name == "y" for name in names)
    if y_count != 1:
        raise ValueError(
            f"Dataset must contain exactly one 'y' label column; found {y_count}: {data_file}"
        )

    feature_fields = [name for name in names if name != "y"]
    if len(feature_fields) != 2:
        raise ValueError(
            f"Dataset must contain exactly two feature columns; found {len(feature_fields)}: {data_file}"
        )
    if set(feature_fields) != {"x1", "x2"}:
        raise ValueError(
            f"Feature columns must be named 'x1' and 'x2'; found {feature_fields}: {data_file}"
        )

    # seperate data into x and y
    x = np.column_stack([raw[name] for name in feature_fields])
    y = np.asarray(raw["y"], dtype=float)

    if y.ndim != 1:
        y = y.reshape(-1)

    # make sure the dataset values are correct
    if np.isnan(y).any():
        raise ValueError(f"Label column 'y' contains NaN values: {data_file}")

    unique_labels = set(np.unique(y))
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            f"Label column 'y' must be binary (0 or 1); found values {sorted(unique_labels)}: {data_file}"
        )

    y = y.reshape(-1, 1)
    y = np.hstack([y, 1 - y])

    # shape (batches, 2) & (batches, 1)
    return (x, y)
