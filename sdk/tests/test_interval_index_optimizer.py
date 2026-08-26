"""Tests for bounded, online interval-index maintenance."""

import pytest

from atriumdb import AtriumSDK


SEC = 1_000_000_000


@pytest.fixture
def sdk(tmp_path):
    dataset = AtriumSDK.create_dataset(dataset_location=tmp_path / "dataset", database_type="sqlite")
    try:
        yield dataset
    finally:
        dataset.close()


def _raw_intervals(sdk, measure_id, device_id):
    with sdk.sql_handler.connection(begin=False) as (conn, cursor):
        cursor.execute(
            "SELECT start_time_n, end_time_n FROM interval_index "
            "WHERE measure_id = ? AND device_id = ? ORDER BY start_time_n, id",
            (measure_id, device_id))
        return [tuple(map(int, row)) for row in cursor.fetchall()]


def test_optimizer_merges_fast_rows_across_bounded_pages(sdk):
    measure_id = sdk.insert_measure("optimizer", freq=1, freq_units="Hz")
    device_id = sdk.insert_device("optimizer-device")
    sdk.sql_handler.insert_intervals([
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 0, "end_time_n": 10},
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 25, "end_time_n": 30},
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 100, "end_time_n": 110},
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 111, "end_time_n": 120},
    ])

    stats = sdk.optimize_interval_index(gap_tolerance=15, batch_size=2)

    assert stats == {"pairs_processed": 1, "rows_examined": 4, "rows_merged": 2}
    assert _raw_intervals(sdk, measure_id, device_id) == [(0, 30), (100, 120)]


def test_optimizer_uses_period_based_smart_default_and_scopes_pair(sdk):
    measure_id = sdk.insert_measure("default-policy", freq=1, freq_units="Hz")
    device_id = sdk.insert_device("default-policy-device")
    other_device_id = sdk.insert_device("other-device")
    sdk.sql_handler.insert_intervals([
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 0, "end_time_n": SEC},
        # The 5-second gap is inside the default 10-period tolerance.
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 6 * SEC, "end_time_n": 7 * SEC},
        # The 21-second gap is deliberately retained.
        {"measure_id": measure_id, "device_id": device_id, "start_time_n": 28 * SEC, "end_time_n": 29 * SEC},
        {"measure_id": measure_id, "device_id": other_device_id, "start_time_n": 0, "end_time_n": SEC},
        {"measure_id": measure_id, "device_id": other_device_id, "start_time_n": 6 * SEC, "end_time_n": 7 * SEC},
    ])

    stats = sdk.optimize_interval_index(measure_id=measure_id, device_id=device_id, batch_size=1)

    assert stats == {"pairs_processed": 1, "rows_examined": 3, "rows_merged": 1}
    assert _raw_intervals(sdk, measure_id, device_id) == [(0, 7 * SEC), (28 * SEC, 29 * SEC)]
    assert _raw_intervals(sdk, measure_id, other_device_id) == [(0, SEC), (6 * SEC, 7 * SEC)]


def test_optimizer_rejects_invalid_scope_and_batch_size(sdk):
    with pytest.raises(ValueError, match="device_id requires measure_id"):
        sdk.optimize_interval_index(device_id=1)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        sdk.optimize_interval_index(batch_size=0)
