import numpy as np
import pandas as pd

def chunkify(df: pd.DataFrame, chunk_size: int, stride: int = 1):
    start = 0
    length = df.shape[0]

    # If DF is smaller than the chunk, return the DF
    if length <= chunk_size:
        return df[:]

    # Producing individual chunks
    dfs = []
    # while start + chunk_size <= length:
    #     dfs.append(df[start:chunk_size + start])
    #     start = start + chunk_size
    for i in range(0, length - chunk_size, stride):
        dfs.append(df[i:i + chunk_size])
    return dfs

def shuffle_concat(lists: list) -> np.array:
    """
    Concatenate N lists into N, shuffling samples by row.
    Args:
        lists (list): List of lists to be concatenated and shuffle.
    """
    A = np.concatenate(lists, axis=0)
    np.random.shuffle(A)
    return A
