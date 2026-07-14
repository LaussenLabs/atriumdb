# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The dataset 'overwrite' setting is the merge conflict policy: how block
merging resolves duplicate timestamps between a new write and existing data.

* 'overwrite' and the legacy default 'ignore' - the new write's values win.
* 'protect' - the existing data's values win.
* 'error' - a merge that would drop conflicting data raises instead.

The policy is enforced where deduplication happens (writes smaller than one
block that merge with an existing block). Overlapping writes of a full block or
more never merge, so both copies are stored regardless of the policy and reads
resolve them with allow_duplicates=False."""

from atriumdb.atrium_sdk import AtriumSDK
import numpy as np
import pytest

from tests.testing_framework import _test_for_both

DB_NAME = 'overwrite_test'


def _setup(db_type, dataset_location, connection_params, overwrite=None):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params,
        overwrite=overwrite)
    measure_id = sdk.insert_measure('signal_1', 1, freq_units="Hz")
    device_id = sdk.insert_device('dev_1')
    return sdk, measure_id, device_id


def _write_base_then_overlap(sdk, measure_id, device_id):
    """Write values 1..8 (t^2), then rewrite t=3..5 with t+5. Returns
    (og_time_data, og_value_data, new_value_data)."""
    og_time_data = np.arange(1, 9, dtype=np.int64)
    og_value_data = og_time_data * og_time_data
    sdk.write_data_easy(measure_id, device_id, og_time_data, og_value_data, 1, time_units='s', freq_units="Hz")

    new_time_data = og_time_data[2:5].copy()
    new_value_data = new_time_data + 5
    sdk.write_data_easy(measure_id, device_id, new_time_data, new_value_data, 1, time_units='s', freq_units="Hz")
    return og_time_data, og_value_data, new_value_data


def test_overwrite_policy_new_values_win():
    _test_for_both(DB_NAME, _test_overwrite_policy)


def _test_overwrite_policy(db_type, dataset_location, connection_params):
    sdk, measure_id, device_id = _setup(db_type, dataset_location, connection_params, overwrite='overwrite')
    og_times, og_values, new_values = _write_base_then_overlap(sdk, measure_id, device_id)

    _, read_times, read_values = sdk.get_data(
        measure_id, int(og_times[0]), int(og_times[-1]) + 1, device_id=device_id, time_units='s')

    # No data loss and no duplicated timestamps: the overlapping write was merged.
    assert np.array_equal(read_times, og_times)

    # The newest write wins on the overlapping timestamps; the rest is untouched.
    expected_values = og_values.copy()
    expected_values[2:5] = new_values
    assert np.array_equal(read_values, expected_values)


def test_legacy_ignore_default_behaves_like_overwrite():
    _test_for_both(DB_NAME + '_ignore', _test_legacy_ignore_default)


def _test_legacy_ignore_default(db_type, dataset_location, connection_params):
    # 'ignore' (the recorded default on virtually every existing dataset)
    # historically meant "skip overlap handling"; in the merged write path it
    # resolves to the default policy: new values win.
    sdk, measure_id, device_id = _setup(db_type, dataset_location, connection_params)  # default -> 'ignore'
    assert sdk.settings_dict['overwrite'] == 'ignore'
    og_times, og_values, new_values = _write_base_then_overlap(sdk, measure_id, device_id)

    _, read_times, read_values = sdk.get_data(
        measure_id, int(og_times[0]), int(og_times[-1]) + 1, device_id=device_id, time_units='s')
    expected_values = og_values.copy()
    expected_values[2:5] = new_values
    assert np.array_equal(read_times, og_times)
    assert np.array_equal(read_values, expected_values)


def test_protect_policy_old_values_win():
    _test_for_both(DB_NAME + '_protect', _test_protect_policy)


def _test_protect_policy(db_type, dataset_location, connection_params):
    sdk, measure_id, device_id = _setup(db_type, dataset_location, connection_params, overwrite='protect')
    og_times, og_values, _ = _write_base_then_overlap(sdk, measure_id, device_id)

    _, read_times, read_values = sdk.get_data(
        measure_id, int(og_times[0]), int(og_times[-1]) + 1, device_id=device_id, time_units='s')

    # Merged with no duplicates, but the existing values were protected.
    assert np.array_equal(read_times, og_times)
    assert np.array_equal(read_values, og_values)

    # A partially overlapping write still contributes its NEW timestamps.
    extra_times = np.arange(7, 11, dtype=np.int64)  # 7, 8 conflict; 9, 10 are new
    extra_values = extra_times + 50
    sdk.write_data_easy(measure_id, device_id, extra_times, extra_values, 1, time_units='s', freq_units="Hz")

    _, read_times, read_values = sdk.get_data(
        measure_id, 1, 11, device_id=device_id, time_units='s')
    assert np.array_equal(read_times, np.arange(1, 11, dtype=np.int64))
    expected = np.concatenate((og_values, extra_values[2:]))
    assert np.array_equal(read_values, expected)


def test_protect_policy_old_values_win_gap_path():
    _test_for_both(DB_NAME + '_protect_gap', _test_protect_policy_gap)


def _test_protect_policy_gap(db_type, dataset_location, connection_params):
    # Same policy through the gap-array merge path (write_segment).
    sdk, measure_id, device_id = _setup(db_type, dataset_location, connection_params, overwrite='protect')
    base = 1_000_000_000_000_000_000
    period = 10 ** 9

    sdk.write_segment(measure_id, device_id, np.arange(10, dtype=np.int64), base, period=period, time_units="ns")
    sdk.write_segment(measure_id, device_id, np.full(4, 99, dtype=np.int64), base + 3 * period,
                      period=period, time_units="ns")

    _, read_times, read_values = sdk.get_data(
        measure_id, base, base + 10 * period + 1, device_id=device_id)
    assert np.array_equal(read_times, base + np.arange(10, dtype=np.int64) * period)
    assert np.array_equal(read_values, np.arange(10, dtype=np.int64))


def test_error_policy_raises_on_conflict():
    _test_for_both(DB_NAME + '_error', _test_error_policy)


def _test_error_policy(db_type, dataset_location, connection_params):
    sdk, measure_id, device_id = _setup(db_type, dataset_location, connection_params, overwrite='error')

    og_time_data = np.arange(1, 9, dtype=np.int64)
    og_value_data = og_time_data * og_time_data
    sdk.write_data_easy(measure_id, device_id, og_time_data, og_value_data, 1, time_units='s', freq_units="Hz")

    # A non-conflicting continuation merges without complaint.
    more_times = np.arange(9, 12, dtype=np.int64)
    sdk.write_data_easy(measure_id, device_id, more_times, more_times, 1, time_units='s', freq_units="Hz")

    # A write sharing timestamps with existing data refuses to merge.
    with pytest.raises(ValueError, match="overwrite setting is 'error'"):
        sdk.write_data_easy(measure_id, device_id, og_time_data[2:5], og_time_data[2:5] + 5,
                            1, time_units='s', freq_units="Hz")

    # Nothing was written by the failed call and the earlier data is intact.
    _, read_times, read_values = sdk.get_data(
        measure_id, 1, 12, device_id=device_id, time_units='s')
    assert np.array_equal(read_times, np.arange(1, 12, dtype=np.int64))
    assert np.array_equal(read_values, np.concatenate((og_value_data, more_times)))


def test_large_overlapping_write_duplicates_data():
    _test_for_both(DB_NAME + '_large', _test_large_overlapping_write)


def _test_large_overlapping_write(db_type, dataset_location, connection_params):
    """A write of at least one full block never merges, so overlapping data is
    stored twice regardless of the policy. This pins the current behavior: no
    error is raised, the read default surfaces the duplicate timestamps, and
    allow_duplicates=False resolves them to one value per timestamp."""
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk.block.block_size = 10

    measure_id = sdk.insert_measure('signal_1', 1, freq_units="Hz")
    device_id = sdk.insert_device('dev_1')

    times = np.arange(1, 21, dtype=np.int64)  # 20 values >= block_size, so no block merge
    values_first = times * times
    values_second = times + 5

    sdk.write_data_easy(measure_id, device_id, times, values_first, 1, time_units='s', freq_units="Hz")
    sdk.write_data_easy(measure_id, device_id, times, values_second, 1, time_units='s', freq_units="Hz")

    # Both copies are on disk; the default read returns every stored sample.
    _, read_times, _ = sdk.get_data(
        measure_id, int(times[0]), int(times[-1]) + 1, device_id=device_id, time_units='s')
    assert read_times.size == 2 * times.size

    # allow_duplicates=False collapses to one value per timestamp.
    _, dedup_times, dedup_values = sdk.get_data(
        measure_id, int(times[0]), int(times[-1]) + 1, device_id=device_id, time_units='s',
        allow_duplicates=False)
    assert np.array_equal(dedup_times, times)
    for t, v in zip(dedup_times, dedup_values):
        idx = int(t) - 1
        assert v in (values_first[idx], values_second[idx])
