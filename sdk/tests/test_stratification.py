import pytest
import numpy as np
import random

from atriumdb.windowing.defintion_splitter import stratified_partition_by_labels


def test_stratified_partition_basic_functionality():
    """
    Test basic functionality with a small dataset.
    """
    data_list = generate_data_list(100)
    partition_ratios = [50, 50]
    # partitions, partition_sums, partition_source_counts
    partitions, label_totals, partition_source_counts = stratified_partition_by_labels(data_list, partition_ratios, random_state=22)

    assert len(partitions) == 2, "Should create two partitions"
    assert all(isinstance(part, list) for part in partitions), "Partitions should be lists"
    assert sum(len(part) for part in partitions) == len(data_list), "Should partition all data"

def test_stratified_partition_large_dataset():
    """
    Test the partitioning on a large dataset and check if ratios are correct within a 10% margin.
    """
    data_list = generate_data_list(100000)
    partition_ratios = [50, 50]
    partitions, label_totals, partition_source_counts = stratified_partition_by_labels(data_list, partition_ratios, random_state=22)

    assert_within_threshold(label_totals[0], label_totals[1], threshold=0.1)


def assert_within_threshold(list1, list2, threshold=0.1):
    assert len(list1) == len(list2), "Lists must be of equal length"

    for i in range(len(list1)):
        value1 = list1[i]
        value2 = list2[i]

        # Calculate the acceptable range
        lower_bound = value1 * (1 - threshold)
        upper_bound = value1 * (1 + threshold)

        # Check if value2 is within the acceptable range
        assert lower_bound <= value2 <= upper_bound, f"Element at index {i} is out of threshold range: {value2} not within [{lower_bound}, {upper_bound}]"


def test_invalid_ratios():
    """
    Test the function with invalid partition ratios.
    """
    data_list = generate_data_list(100)
    partition_ratios = [0, 0]  # Invalid ratio
    with pytest.raises(ValueError):
        stratified_partition_by_labels(data_list, partition_ratios, random_state=22)



def generate_data_list(num_entries):
    data_list = []

    # Define the possible values for the first column
    first_column_values = ["patient_ids", "device_ids", "device_patient_tuples"]

    # Create the specified number of entries
    for _ in range(num_entries):
        # Randomly select values for the first and second columns
        first_column = random.choice(first_column_values)
        if first_column == "device_patient_tuples":
            second_column = (random.randint(1, 10), random.randint(1, 10))  # Generate a tuple of two IDs
        else:
            second_column = random.randint(1, 10)  # Generate a single ID

        # Generate random nanosecond durations for the remaining columns
        durations = [random.randint(1, 10000000000) for _ in range(3)]

        # Append the row to the data_list
        data_list.append([first_column, second_column, [[]]] + durations)

    return data_list
