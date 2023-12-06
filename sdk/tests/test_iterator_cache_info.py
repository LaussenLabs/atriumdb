import pytest

# Mock a minimal representation of the SDK and other required objects
from atriumdb import DatasetIterator


class MockSDK:
    pass


def test_dataset_iterator_extract_cache_info():
    sdk = MockSDK()
    validated_measure_list = []
    validated_label_set_list = []
    validated_sources = {
        'device_ids': {
            1: [(0, 100)],
        },
    }
    window_duration_ns = 30
    window_slide_ns = 40
    num_windows_prefetch = 2

    # Create an instance of the DatasetIterator with those mock parameters
    iterator = DatasetIterator(
        sdk=sdk,
        validated_measure_list=validated_measure_list,
        validated_label_set_list=validated_label_set_list,
        validated_sources=validated_sources,
        window_duration_ns=window_duration_ns,
        window_slide_ns=window_slide_ns,
        num_windows_prefetch=num_windows_prefetch
    )

    # Use the private method _extract_cache_info and check if it returns the correct cache information
    cache_info, starting_window_index_per_batch, total_num_windows = iterator._extract_cache_info()

    # Define expected outcomes based on the mocked data and initialization parameters.
    expected_cache_info = [
        [['device_ids', 1, 0, 70, 0, 70, 2]],
        [['device_ids', 1, 80, 110, 80, 100, 1]]
    ]
    expected_starting_window_index_per_batch = [0, 2, 3]
    expected_total_num_windows = 3

    # Verify that the generated cache information matches the expectations
    assert cache_info == expected_cache_info
    assert starting_window_index_per_batch == expected_starting_window_index_per_batch
    assert total_num_windows == expected_total_num_windows
