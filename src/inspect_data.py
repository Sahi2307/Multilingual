import numpy as np
from pathlib import Path

def inspect():
    data_dir = Path("data/processed")
    npz_path = data_dir / "urgency_features.npz"
    data = np.load(npz_path)

    y_train = data["y_train"].astype(int)
    y_val = data["y_val"].astype(int)
    y_test = data["y_test"].astype(int)
    X_train = data["X_train"]

    print(f"X_train shape: {X_train.shape}")
    print(f"Train counts: {np.bincount(y_train)}")
    print(f"Val counts: {np.bincount(y_val)}")
    print(f"Test counts: {np.bincount(y_test)}")

if __name__ == "__main__":
    inspect()
