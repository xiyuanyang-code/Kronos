import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    FILES = ["./data/test_X.npy", "./data/val_X.npy","./data/train_X.npy"]
    # FILES = ["./data/test_X.npy", "./data/val_X.npy", "./data/train_X.rar"]
    for file in FILES:
        numpy_array: np.ndarray = np.load(file, allow_pickle=True)
        print(numpy_array.ndim)
        print(numpy_array.shape)
        # print(numpy_array)
