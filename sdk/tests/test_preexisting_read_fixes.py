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
"""
Regression guards for GENERAL (non-feature, pre-existing) read-side defects found
while reviewing the aperiodic + text feature. Each of these reproduces identically
on ``main`` with plain numeric waveform measures -- they are NOT regressions of the
aperiodic feature, and the tests below are written with waveform measures only so
they keep their meaning independent of that feature.

Every test asserts the FIXED behaviour and must PASS. SQLite only.
"""
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC


@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_preexisting_read_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def _defn(sdk, tags, device_id, start_ns, end_ns):
    d = DatasetDefinition(
        measures=list(tags),
        device_ids={device_id: [{"start": int(start_ns), "end": int(end_ns)}]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d.validate(sdk)
    return d


def _mixed_rate_dataset(sdk, fast_hz=250.0, slow_hz=1.0, seconds=20):
    dev = sdk.insert_device(device_tag="d1")
    m_fast = sdk.insert_measure("ECG", freq=fast_hz, freq_units="Hz", units="mV")
    m_slow = sdk.insert_measure("slow", freq=slow_hz, freq_units="Hz", units="mV")
    n_fast = int(fast_hz * seconds)
    sdk.write_time_value_pairs(
        m_fast, dev, BASE + (np.arange(n_fast, dtype=np.int64) * int(SEC / fast_hz)),
        np.arange(float(n_fast)))
    n_slow = int(slow_hz * seconds)
    sdk.write_time_value_pairs(
        m_slow, dev, BASE + (np.arange(n_slow, dtype=np.int64) * int(SEC / slow_hz)),
        np.arange(float(n_slow)))
    return dev


# --------------------------------------------------------------------------- #
# A. the default prefetch batch must be sized by the FASTEST measure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shuffle", [False, True])
def test_default_prefetch_batch_is_not_inflated_by_a_slow_measure(sdk, shuffle):
    """Adding a low-rate measure must not inflate the default prefetch batch.

    ``num_windows_prefetch`` was derived from ``min(freq_nhz)`` (the SLOWEST
    measure) while the iterator allocates ``max_batch_size * row_size`` values and
    ``row_size`` comes from the LOWEST period (the FASTEST measure). Adding one
    1 Hz measure to a 250 Hz / 10 s-window definition therefore took the default
    batch from ~10 MB to ~2621 MB (~26 GB with shuffle enabled). The batch is now
    sized by the fastest measure, so a slow companion measure costs only its own
    (small) array.
    """
    dev = _mixed_rate_dataset(sdk)

    def batch_values(tags):
        d = _defn(sdk, tags, dev, BASE, BASE + 20 * SEC)
        it = sdk.get_iterator(d, 10 * SEC, 10 * SEC, shuffle=shuffle)
        return it.max_batch_size * it.row_size

    fast_only = batch_values(["ECG"])
    mixed = batch_values(["ECG", "slow"])

    assert mixed == fast_only, (
        f"adding a 1 Hz measure inflated the prefetch batch from {fast_only} to "
        f"{mixed} values ({mixed / max(fast_only, 1):.0f}x)")
    # sanity: the fast-only default is itself a sane size (tens of MB, not GB)
    assert fast_only * 8 < 512 * 10 ** 6


def test_prefetch_batch_default_is_unchanged_for_a_single_rate_definition(sdk):
    """min == max for a single-rate definition, so the numeric path is untouched."""
    dev = _mixed_rate_dataset(sdk)
    d = _defn(sdk, ["ECG"], dev, BASE, BASE + 20 * SEC)
    it = sdk.get_iterator(d, 10 * SEC, 10 * SEC, shuffle=False)
    # 10 * block_size // (values per 10 s slide at 250 Hz)
    assert it.row_size == 2500
    assert it.max_batch_size == (10 * sdk.block.block_size) // 2500


# --------------------------------------------------------------------------- #
# B. a window slide wider than one block must not floor the batch size to zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shuffle", [False, True])
def test_wide_window_slide_does_not_floor_the_batch_size_to_zero(sdk, shuffle):
    """``block_size // values_per_slide`` hits 0 once a slide holds more values
    than one block (e.g. 600 s at 250 Hz), and ``100 * 0 == 0`` left the iterator
    with ``max_batch_size == 0`` -- degenerating every batch to a single window.
    An explicitly supplied ``cached_windows_per_source`` is already asserted > 0;
    the derived default must be too."""
    dev = _mixed_rate_dataset(sdk, seconds=20)
    d = _defn(sdk, ["ECG"], dev, BASE, BASE + 20 * SEC)
    it = sdk.get_iterator(d, 600 * SEC, 600 * SEC, shuffle=shuffle)
    assert it.max_batch_size >= 1


def test_windows_are_unchanged_by_the_new_batch_sizing(sdk):
    """The batch size is a pure RAM knob: the emitted windows must be identical to
    an explicit single-batch run."""
    dev = _mixed_rate_dataset(sdk, seconds=20)
    d = _defn(sdk, ["ECG", "slow"], dev, BASE, BASE + 20 * SEC)

    def run(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = []
            for w in sdk.get_iterator(d, 5 * SEC, 5 * SEC, shuffle=False, **kw):
                for key in sorted(w.signals, key=lambda k: k[0]):
                    v = np.asarray(w.signals[key]["values"], dtype=np.float64)
                    out.append((int(w.start_time), key[0], np.nan_to_num(v, nan=-9e9).tobytes()))
            return out

    assert run() == run(num_windows_prefetch=1)
    assert run() == run(num_windows_prefetch=1000)


# =========================================================================== #
# READ-SIDE DUPLICATE HANDLING
#
# Write-path dedup is a best-effort side effect of the small-write block merge (write
# speed is the priority and duplicates are expected at live ingest), so surviving
# duplicates are resolved on READ via ``allow_duplicates=False``.
#
# Semantics under test:
#   * a duplicate is two samples with the SAME TIMESTAMP, whether or not the values agree;
#   * exactly one sample per timestamp is returned;
#   * the survivor follows the dataset's ``overwrite`` merge conflict policy -- newest
#     write wins by default, existing value wins under 'protect' -- so a read resolves a
#     duplicate the same way a write would have;
#   * the default (``allow_duplicates=True``) is unchanged and returns every stored sample.
# =========================================================================== #

DUP_BLOCK = 8            # tiny, so a duplicate write never takes the merge/dedup path
DUP_N = 20
DUP_WINDOW = (int(BASE), int(BASE) + DUP_N * SEC)


def _duplicated_dataset(sdk, offsets=(0.0, 1000.0)):
    """Write the same DUP_N timestamps once per offset. Each write is >= block_size so it
    never merges, which is exactly how a replayed live-ingest buffer lands on disk."""
    sdk.block.block_size = DUP_BLOCK
    dev = sdk.insert_device(device_tag="dup_dev")
    measure_id = sdk.insert_measure("ECG", freq=1, freq_units="Hz", units="mV")
    times = BASE + np.arange(DUP_N, dtype=np.int64) * SEC
    for offset in offsets:
        sdk.write_time_value_pairs(
            measure_id, dev, times, np.arange(DUP_N, dtype=np.float64) + offset, period=SEC)
    return measure_id, dev, times


def test_read_default_still_returns_every_duplicate(sdk):
    """The default must not change: every stored sample comes back, duplicates included."""
    measure_id, dev, times = _duplicated_dataset(sdk)
    _, r_times, r_values = sdk.get_data(measure_id, *DUP_WINDOW, device_id=dev)
    assert r_times.size == 2 * DUP_N
    assert r_values.size == 2 * DUP_N


def test_allow_duplicates_false_collapses_to_one_sample_per_timestamp(sdk):
    """One sample per timestamp, sorted, and the most recently written value survives
    under the default 'overwrite'/'ignore' merge conflict policy."""
    measure_id, dev, times = _duplicated_dataset(sdk)
    _, r_times, r_values = sdk.get_data(measure_id, *DUP_WINDOW, device_id=dev,
                                        allow_duplicates=False)
    assert np.array_equal(r_times, times)
    assert np.unique(r_times).size == r_times.size
    assert np.array_equal(r_values, np.arange(DUP_N, dtype=np.float64) + 1000.0)


def test_duplicate_keep_first_selects_the_earliest_written_copy(sdk):
    """``duplicate_keep`` overrides the policy-derived survivor per call."""
    measure_id, dev, times = _duplicated_dataset(sdk)
    _, r_times, r_values = sdk.get_data(measure_id, *DUP_WINDOW, device_id=dev,
                                        allow_duplicates=False, duplicate_keep="first")
    assert np.array_equal(r_times, times)
    assert np.array_equal(r_values, np.arange(DUP_N, dtype=np.float64))

    with pytest.raises(ValueError, match="duplicate_keep"):
        sdk.get_data(measure_id, *DUP_WINDOW, device_id=dev, allow_duplicates=False,
                     duplicate_keep="oldest")


def test_protect_policy_keeps_the_existing_value_on_read():
    """Under ``overwrite='protect'`` the write path keeps existing values, so the read-side
    collapse must keep them too -- read and write cannot disagree."""
    loc = tempfile.mkdtemp(prefix="atrium_preexisting_read_protect_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite",
                                 overwrite="protect")
    try:
        measure_id, dev, times = _duplicated_dataset(s)
        _, r_times, r_values = s.get_data(measure_id, *DUP_WINDOW, device_id=dev,
                                          allow_duplicates=False)
        assert np.array_equal(r_times, times)
        assert np.array_equal(r_values, np.arange(DUP_N, dtype=np.float64))
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def test_allow_duplicates_false_on_string_reads(sdk):
    """``get_string_data`` collapses on the stored codes before decoding, so the surviving
    text is the surviving sample's own."""
    sdk.block.block_size = DUP_BLOCK
    dev = sdk.insert_device(device_tag="dup_str_dev")
    measure_id = sdk.insert_measure("alarm", freq=1, freq_units="Hz", units="string",
                                    signal_kind="event", value_type="string")
    times = BASE + np.arange(DUP_N, dtype=np.int64) * SEC
    sdk.write_time_value_pairs(measure_id, dev, times,
                               np.array(["OLD"] * DUP_N, dtype=object), period=SEC)
    sdk.write_time_value_pairs(measure_id, dev, times,
                               np.array(["NEW"] * DUP_N, dtype=object), period=SEC)

    dup_times, dup_values = sdk.get_string_data(measure_id, *DUP_WINDOW, device_id=dev)
    assert dup_times.size == 2 * DUP_N

    r_times, r_values = sdk.get_string_data(measure_id, *DUP_WINDOW, device_id=dev,
                                            allow_duplicates=False)
    assert np.array_equal(r_times, times)
    assert list(r_values) == ["NEW"] * DUP_N
    assert set(dup_values) == {"OLD", "NEW"}, "both texts are genuinely stored"


def test_duplicate_collapse_keeps_distinct_timestamps_untouched(sdk):
    """Do-no-harm: collapsing must only ever remove a repeat of a timestamp already
    returned. Non-duplicated data is byte-identical either way."""
    sdk.block.block_size = DUP_BLOCK
    dev = sdk.insert_device(device_tag="clean_dev")
    measure_id = sdk.insert_measure("ECG", freq=1, freq_units="Hz", units="mV")
    times = BASE + np.arange(3 * DUP_N, dtype=np.int64) * SEC
    values = np.arange(3 * DUP_N, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, dev, times, values, period=SEC)

    window = (int(BASE), int(BASE) + 3 * DUP_N * SEC)
    _, plain_t, plain_v = sdk.get_data(measure_id, *window, device_id=dev)
    _, dedup_t, dedup_v = sdk.get_data(measure_id, *window, device_id=dev,
                                       allow_duplicates=False)
    assert plain_t.tobytes() == dedup_t.tobytes()
    assert plain_v.tobytes() == dedup_v.tobytes()
