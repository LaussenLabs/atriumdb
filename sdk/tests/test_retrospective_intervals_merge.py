import numpy as np

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both


DB_NAME = "atrium-intervals-merge"


def test_merge_all_intervals():
    _test_for_both(DB_NAME, _test_merge_all_intervals)


def _test_merge_all_intervals(db_type, dataset_location, connection_params):
    # Create an SDK instance
    sdk = AtriumSDK.create_dataset(dataset_location=dataset_location, database_type=db_type,
                                   connection_params=connection_params)

    # Define different scenarios for measure and device pairs
    measure_device_pairs = [
        ("ECG Lead I", "Monitor A1"),  # measure-device pair with no intervals
        ("ECG Lead II", "Monitor A2"),  # measure-device pair with intervals, but none to merge
        ("ECG Lead III", "Monitor A3"),  # measure-device pair with intervals that need to be merged
        ("ECG Lead IV", "Monitor A4"),
    ]

    # Define the intervals for each scenario
    intervals_scenarios = [
        np.array([], dtype=np.int64),  # no intervals
        np.array([(1, 2), (3, 4), (5, 6)], dtype=np.int64),  # intervals, but none to merge
        np.array([(1, 2), (2, 3), (2, 3), (3, 5), (4, 5), (5, 7), (6, 8), (9, 10)], dtype=np.int64)
        # intervals that need to be merged
    ]

    # Define the expected results after merging
    expected_results = [
        np.array([], dtype=np.int64),  # no intervals to merge
        np.array([(1, 2), (3, 4), (5, 6)], dtype=np.int64),  # intervals remain the same, no merge needed
        np.array([(1, 8), (9, 10)], dtype=np.int64),  # merged intervals
    ]

    # Randomly generate hundreds of intervals and add as a scenario
    random_intervals = np.array([(i, i + 2) for i in range(1, 200, 2)], dtype=np.int64)
    np.random.shuffle(random_intervals)
    intervals_scenarios.append(random_intervals)

    # We expect all intervals to be merged into one large interval for the random scenario
    expected_results.append(np.array([(1, 201)], dtype=np.int64))

    # Insert measures and devices and intervals
    for (measure_tag, device_tag), intervals, expected in zip(measure_device_pairs, intervals_scenarios,
                                                              expected_results):
        measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=500, units="mV", freq_units="Hz")
        device_id = sdk.insert_device(device_tag=device_tag)

        # Insert intervals using `insert_interval_array`
        if intervals.size != 0:
            sdk.insert_interval_array(measure_id=measure_id, device_id=device_id, interval_array=intervals)

    # Merge Intervals
    sdk.merge_all_intervals()

    # Test Each Scenario
    for (measure_tag, device_tag), intervals, expected in zip(measure_device_pairs, intervals_scenarios,
                                                              expected_results):
        measure_id = sdk.get_measure_id(measure_tag=measure_tag, freq=500, units="mV", freq_units="Hz")
        device_id = sdk.get_device_id(device_tag=device_tag)

        # Fetch the current intervals after merging using `get_interval_array`
        current_intervals = sdk.get_interval_array(measure_id=measure_id, device_id=device_id)

        # Assertion to check if the intervals are merged as expected
        np.testing.assert_array_equal(current_intervals, expected,
                                      err_msg=f"Expected {expected} but got {current_intervals}")
