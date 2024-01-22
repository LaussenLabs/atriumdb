import numpy as np


def get_threshold_labels(sliced_labels, label_threshold=0.5):
    # Calculate the percentage of 1s for each label in each window
    percentages = np.mean(sliced_labels, axis=-1)
    # Apply threshold
    return (percentages > label_threshold).astype(int)
