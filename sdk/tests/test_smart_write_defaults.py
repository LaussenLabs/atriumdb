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
Tests for the "smart write defaults" feature: when the caller does not specify a
gap_tolerance or a time encoding, AtriumDB picks sensible defaults from the data,
and a `continuous` flag forces a single interval. Covers write_data,
write_segments and write_time_value_pairs, buffered and non-buffered.

The SDK-level tests run against both sqlite and MariaDB (the MariaDB runs are
skipped when no server is configured in .env).
"""
import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest
from dotenv import load_dotenv

from atriumdb import AtriumSDK
from atriumdb.adb_functions import (
    choose_interval_gap_tolerance,
    choose_time_encoding,
    collapse_continuous_write_intervals,
    observed_median_delta_ns,
    observed_median_delta_from_gap_array,
    widen_gap_tolerance_for_observed_spacing,
    reencode_dataset,
    APERIODIC_MIN_PERIOD_NS,
    DEFAULT_GAP_TOLERANCE_CEILING_NS,
    DEFAULT_GAP_TOLERANCE_FLOOR_NS,
    DEFAULT_GAP_TOLERANCE_PERIODS,
    ENCODE_RAW_GAP_FLOOR,
)
from atriumdb.helpers.block_constants import TIME_TYPES, COMPRESSION_TYPES

SEC = 1_000_000_000
GAP = TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']  # 2
TS = TIME_TYPES['TIME_ARRAY_INT64_NS']                 # 1
NONE = COMPRESSION_TYPES['NONE']                       # 1
ZSTD = COMPRESSION_TYPES['ZSTD']                       # 3

def _rng():
    """Fresh generator per call site, so every test gets identical data
    regardless of execution order or fixture parametrization (a shared stateful
    generator would hand different draws to the sqlite and mariadb runs of the
    same test)."""
    return np.random.default_rng(2024)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
MARIA_DB_NAME = "smart_write_defaults_test"


@pytest.fixture(params=["sqlite", "mariadb"])
def sdk(request):
    loc = tempfile.mkdtemp(prefix="atrium_swd_")
    shutil.rmtree(loc, ignore_errors=True)
    if request.param == "sqlite":
        s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    else:
        load_dotenv()
        host = os.getenv("MARIA_DB_HOST")
        if host is None:
            pytest.skip("MariaDB connection not configured (.env)")
        from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
        connection_params = {'sqltype': 'mariadb', 'host': host, 'user': os.getenv("MARIA_DB_USER"),
                             'password': os.getenv("MARIA_DB_PASSWORD"), 'database': MARIA_DB_NAME,
                             'port': int(os.getenv("MARIA_DB_PORT"))}
        maria_handler = MariaDBHandler(host, connection_params['user'], connection_params['password'],
                                       MARIA_DB_NAME, connection_params['port'])
        drop_conn = maria_handler.maria_connect_no_db()
        drop_conn.cursor().execute(f"DROP DATABASE IF EXISTS `{MARIA_DB_NAME}`")
        drop_conn.close()
        s = AtriumSDK.create_dataset(dataset_location=loc, database_type="mariadb",
                                     connection_params=connection_params)
    try:
        yield s
    finally:
        # Close the pooled MariaDB connection: the whole suite runs in one
        # process, and leaking one connection per test exhausts max_connections.
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def new_md(sdk, freq_hz):
    """Create a fresh measure+device for a given frequency."""
    m = sdk.insert_measure(measure_tag=f"m_{new_md.counter}", freq=float(freq_hz), freq_units="Hz")
    d = sdk.insert_device(device_tag=f"d_{new_md.counter}")
    new_md.counter += 1
    return m, d


new_md.counter = 0


def read_back(sdk, m, d, times, period_ns):
    end = int(times[-1]) + int(period_ns) + 1
    return sdk.get_data(m, int(times[0]), end, device_id=d)


def n_interval_rows(sdk, m, d):
    iv = sdk.get_interval_array(m, device_id=d, gap_tolerance_nano=0)
    return len(iv)


def encodings(headers):
    return set((int(h.t_encoded_type), int(h.t_compression)) for h in headers)


def to_segments(times, values, period_ns):
    """Split a timestamp array into contiguous runs (each internally regular at
    period_ns) so the same data can be written via write_segments."""
    boundaries = np.nonzero(np.diff(times) != period_ns)[0] + 1
    chunks = np.split(np.arange(times.size), boundaries) if times.size else []
    segs = [values[c].copy() for c in chunks]
    starts = [int(times[c[0]]) for c in chunks]
    return segs, starts


# --------------------------------------------------------------------------- #
# Scenario generators -> (times_ns int64, values int64, period_ns)
# These mirror the data shapes real adapters / researcher exports produce.
# --------------------------------------------------------------------------- #
def sc_regular_waveform(n=4000, freq_hz=250):
    period = SEC // freq_hz
    t = 1_600_000_000 * SEC + np.arange(n, dtype=np.int64) * period
    return t, np.arange(n, dtype=np.int64), period


def sc_waveform_rare_resync(n=4000, freq_hz=250, n_resync=4):
    t, v, period = sc_regular_waveform(n, freq_hz)
    idx = np.sort(_rng().choice(np.arange(1, n), size=n_resync, replace=False))
    for i in idx:
        t[i:] += 7 * 1_000_000  # 7 ms clock jump
    return t, v, period


def sc_waveform_msg_jitter(freq_hz=125, msg=120, n_msg=200, jitter_ms=2):
    rng = _rng()
    period = SEC // freq_hz
    n = msg * n_msg
    t = np.empty(n, dtype=np.int64)
    cur = 1_600_000_000 * SEC
    for k in range(n_msg):
        t[k * msg:(k + 1) * msg] = cur + np.arange(msg, dtype=np.int64) * period
        cur += msg * period + int(rng.integers(-jitter_ms, jitter_ms + 1)) * 1_000_000
    return t, np.arange(n, dtype=np.int64), period


def sc_waveform_float_time(n=4000, freq_hz=128):
    period = SEC // freq_hz
    secs = 1_600_000_000.0 + np.arange(n) / freq_hz
    t = np.round(secs * 1e9).astype(np.int64)
    return t, np.arange(n, dtype=np.int64), period


def sc_metric_1hz(n=1200):
    period = SEC
    return 1_600_000_000 * SEC + np.arange(n, dtype=np.int64) * period, np.arange(n, dtype=np.int64), period


def sc_metric_1_to_8s(n=1500):
    deltas = _rng().integers(1, 9, size=n - 1).astype(np.int64) * SEC
    t = np.empty(n, dtype=np.int64)
    t[0] = 1_600_000_000 * SEC
    t[1:] = t[0] + np.cumsum(deltas)
    return t, np.arange(n, dtype=np.int64), SEC


def sc_metric_aperiodic(n=1500, mean_s=2.0):
    deltas = np.maximum(1, np.round(_rng().exponential(mean_s, size=n - 1) * 1e9)).astype(np.int64)
    t = np.empty(n, dtype=np.int64)
    t[0] = 1_600_000_000 * SEC
    t[1:] = t[0] + np.cumsum(deltas)
    vals, counts = np.unique(deltas, return_counts=True)
    return t, np.arange(n, dtype=np.int64), int(vals[counts.argmax()])


def sc_metric_early_late(n=1500, period_s=2, jitter_ms=200):
    rng = _rng()
    period = period_s * SEC
    t = np.empty(n, dtype=np.int64)
    t[0] = 1_600_000_000 * SEC
    cur = t[0]
    for i in range(1, n):
        step = period * (int(rng.integers(2, 6)) if rng.random() < 0.02 else 1)
        cur += step + int(rng.integers(-jitter_ms, jitter_ms + 1)) * 1_000_000
        t[i] = cur
    return t, np.arange(n, dtype=np.int64), period


def sc_metric_10min(n=200, jitter_s=3):
    rng = _rng()
    period = 600 * SEC
    t = np.empty(n, dtype=np.int64)
    t[0] = 1_600_000_000 * SEC
    cur = t[0]
    for i in range(1, n):
        cur += period + int(rng.integers(-jitter_s, jitter_s + 1)) * SEC
        t[i] = cur
    return t, np.arange(n, dtype=np.int64), period


def sc_bounded_random_50pct(n=50000, freq_hz=250):
    """250 Hz base where ~50% of deltas get a random sub-period offset. Here a
    compressed gap array is still smaller than a compressed timestamp array even
    though the (ns-resolution) offsets are nearly all distinct, so the encoding
    choice must rely on measured sizes rather than the distinct-ratio heuristic."""
    rng = _rng()
    period = SEC // freq_hz
    deltas = np.full(n - 1, period, dtype=np.int64)
    k = (n - 1) // 2
    idx = rng.choice(n - 1, size=k, replace=False)
    deltas[idx] = period + rng.integers(1, period, size=k)
    t = np.empty(n, dtype=np.int64)
    t[0] = 1_600_000_000 * SEC
    t[1:] = t[0] + np.cumsum(deltas)
    return t, np.arange(n, dtype=np.int64), period


def sc_metric_hourly(n=120, jitter_s=10):
    rng = _rng()
    period = 3600 * SEC
    t = np.empty(n, dtype=np.int64)
    t[0] = 1_600_000_000 * SEC
    cur = t[0]
    for i in range(1, n):
        cur += period + int(rng.integers(-jitter_s, jitter_s + 1)) * SEC
        t[i] = cur
    return t, np.arange(n, dtype=np.int64), period


ALL_SCENARIOS = {
    "regular_waveform": sc_regular_waveform,
    "waveform_rare_resync": sc_waveform_rare_resync,
    "waveform_msg_jitter": sc_waveform_msg_jitter,
    "waveform_float_time": sc_waveform_float_time,
    "metric_1hz": sc_metric_1hz,
    "metric_1_to_8s": sc_metric_1_to_8s,
    "metric_aperiodic": sc_metric_aperiodic,
    "metric_early_late": sc_metric_early_late,
    "metric_10min": sc_metric_10min,
    "metric_hourly": sc_metric_hourly,
}


# --------------------------------------------------------------------------- #
# 1. Pure-function unit tests for the default-choosing helpers (no SDK / codec)
# --------------------------------------------------------------------------- #
class TestChooseIntervalGapTolerance:
    def test_metric_uses_period_multiple(self):
        # 1 Hz metric -> 10 * 1s = 10s dominates the 200ms floor.
        assert choose_interval_gap_tolerance(SEC) == DEFAULT_GAP_TOLERANCE_PERIODS * SEC

    def test_waveform_uses_floor(self):
        # 250 Hz -> 10 * 4ms = 40ms < 200ms floor, so the floor wins.
        period = SEC // 250
        assert choose_interval_gap_tolerance(period) == DEFAULT_GAP_TOLERANCE_FLOOR_NS

    def test_slow_metric_scales_with_period(self):
        period = 600 * SEC
        assert choose_interval_gap_tolerance(period) == 10 * period

    def test_none_period_returns_floor(self):
        assert choose_interval_gap_tolerance(None) == DEFAULT_GAP_TOLERANCE_FLOOR_NS

    def test_absurd_period_is_clamped_to_ceiling(self):
        # e.g. a frequency accidentally declared in the wrong unit (1 nHz ->
        # period 10**18 ns); 10x that would overflow the SQL BIGINT tolerance.
        assert choose_interval_gap_tolerance(10 ** 18) == DEFAULT_GAP_TOLERANCE_CEILING_NS

    def test_widening_is_clamped_to_ceiling(self):
        assert widen_gap_tolerance_for_observed_spacing(10 * SEC, 10 ** 18) == DEFAULT_GAP_TOLERANCE_CEILING_NS


class TestCollapseContinuousWriteIntervals:
    def test_collapses_own_gaps_to_one_interval(self):
        intervals = [[0, 5 * SEC], [100 * SEC, 110 * SEC]]
        assert collapse_continuous_write_intervals(intervals, 0, 110 * SEC) == [[0, 110 * SEC]]

    def test_single_point_write_stays_bounded(self):
        # A zero-span write collapses to its own bounds, nothing more.
        assert collapse_continuous_write_intervals([[7 * SEC, 7 * SEC]], 7 * SEC, 7 * SEC) == [[7 * SEC, 7 * SEC]]

    def test_merged_old_data_outside_bounds_keeps_own_rows(self):
        # After a block merge the intervals can include old data lying wholly
        # before/after the caller's own span; continuity is not asserted there.
        intervals = [[0, 5 * SEC], [50 * SEC, 55 * SEC], [60 * SEC, 65 * SEC], [200 * SEC, 205 * SEC]]
        out = collapse_continuous_write_intervals(intervals, 50 * SEC, 65 * SEC)
        assert out == [[0, 5 * SEC], [50 * SEC, 65 * SEC], [200 * SEC, 205 * SEC]]


class TestObservedSpacing:
    def test_median_of_regular_waveform_is_period(self):
        t, v, p = sc_regular_waveform()
        assert observed_median_delta_ns(t) == p

    def test_median_of_aperiodic_reflects_arrival_rate(self):
        t, v, p = sc_metric_aperiodic(mean_s=2.0)
        med = observed_median_delta_ns(t)
        # exponential(2s): median = 2s * ln(2) ~ 1.39s - well above the 5 Hz cutoff
        assert APERIODIC_MIN_PERIOD_NS < med < 3 * SEC

    def test_too_few_samples_is_none(self):
        assert observed_median_delta_ns(np.array([5], dtype=np.int64)) is None

    def test_widen_ignores_waveform_spacing(self):
        # median at waveform rates (jitter/float noise) must not widen anything
        assert widen_gap_tolerance_for_observed_spacing(10 * SEC, 4_000_000) == 10 * SEC
        assert widen_gap_tolerance_for_observed_spacing(10 * SEC, None) == 10 * SEC

    def test_widen_uses_cluster_spacing_for_slow_signals(self):
        # 30s typical spacing: gaps under 10x that are "inside the cluster"
        assert widen_gap_tolerance_for_observed_spacing(10 * SEC, 30 * SEC) == \
            DEFAULT_GAP_TOLERANCE_PERIODS * 30 * SEC

    def test_widen_never_shrinks(self):
        assert widen_gap_tolerance_for_observed_spacing(1000 * SEC, 30 * SEC) == 1000 * SEC

    def test_gap_array_median_matches_timestamp_median(self):
        # The same data must yield the same observed spacing whether it arrives
        # as a timestamp array or a gap array (so segments and time-value pairs
        # get the same interval gap tolerance).
        t, v, p = sc_metric_aperiodic(mean_s=2.0)
        deltas = np.diff(t)
        gap_positions = np.nonzero(deltas != p)[0]
        gap_arr = np.empty(2 * gap_positions.size, dtype=np.int64)
        gap_arr[0::2] = gap_positions + 1
        gap_arr[1::2] = deltas[gap_positions] - p
        assert observed_median_delta_from_gap_array(gap_arr, t.size, p) == observed_median_delta_ns(t)

    def test_gap_array_median_too_few_samples_is_none(self):
        assert observed_median_delta_from_gap_array(np.array([], dtype=np.int64), 1, SEC) is None


class TestChooseTimeEncoding:
    def test_regular_is_raw_gap(self):
        t, v, p = sc_regular_waveform()
        assert choose_time_encoding(p, times_ns=t, num_values=t.size) == (GAP, NONE, 0)

    def test_few_gaps_stay_raw(self):
        t, v, p = sc_waveform_rare_resync(n_resync=ENCODE_RAW_GAP_FLOOR - 10)
        et, tc, _ = choose_time_encoding(p, times_ns=t, num_values=t.size)
        assert (et, tc) == (GAP, NONE)

    def test_structured_moderate_is_compressed_gap(self):
        t, v, p = sc_waveform_msg_jitter()
        et, tc, _ = choose_time_encoding(p, times_ns=t, num_values=t.size)
        assert (et, tc) == (GAP, ZSTD)

    def test_float_time_is_compressed_gap(self):
        t, v, p = sc_waveform_float_time()
        et, tc, _ = choose_time_encoding(p, times_ns=t, num_values=t.size)
        assert (et, tc) == (GAP, ZSTD)  # structured tiny deviations compress well

    def test_aperiodic_is_compressed_timestamp(self):
        t, v, p = sc_metric_aperiodic()
        et, tc, _ = choose_time_encoding(p, times_ns=t, num_values=t.size)
        assert (et, tc) == (TS, ZSTD)

    def test_aperiodic_without_timestamp_allowance_uses_gap(self):
        t, v, p = sc_metric_aperiodic()
        et, tc, _ = choose_time_encoding(p, times_ns=t, num_values=t.size, allow_timestamp=False)
        assert (et, tc) == (GAP, ZSTD)


# --------------------------------------------------------------------------- #
# 2. Round-trip correctness across every scenario, every write method,
#    buffered and non-buffered. This is the central correctness guarantee:
#    whatever encoding/tolerance is auto-chosen, the data reads back exactly.
# --------------------------------------------------------------------------- #
def _freq_for(period_ns):
    # frequency that the measure is registered with (10^18 / period)
    return (10 ** 18) / period_ns / SEC  # in Hz


# ~200s SQLite-only for the full scenario x method x buffered matrix -- `slow` keeps
# it out of the sub-5-minute inner loop. It still runs in every full run, unchanged.
@pytest.mark.slow
@pytest.mark.parametrize("scenario", list(ALL_SCENARIOS))
@pytest.mark.parametrize("method", ["time_value_pairs", "segments"])
@pytest.mark.parametrize("buffered", [False, True])
def test_roundtrip_all(sdk, scenario, method, buffered):
    times, values, period = ALL_SCENARIOS[scenario]()
    m, d = new_md(sdk, _freq_for(period))

    def do_writes():
        if method == "time_value_pairs":
            sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")
        else:
            segs, starts = to_segments(times, values, period)
            for seg, st in zip(segs, starts):
                sdk.write_segment(m, d, seg, st, period=period, time_units="ns")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if buffered:
            with sdk.write_buffer():
                do_writes()
        else:
            do_writes()

        _, rt, rv = read_back(sdk, m, d, times, period)

    np.testing.assert_array_equal(rt, times, err_msg=f"{scenario}/{method}/buffered={buffered} times")
    np.testing.assert_array_equal(rv, values, err_msg=f"{scenario}/{method}/buffered={buffered} values")


# --------------------------------------------------------------------------- #
# 3. The smart gap-tolerance default tames interval-index growth.
#    With tolerance 0 the index explodes; with the smart default it collapses.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", ["waveform_float_time", "waveform_msg_jitter",
                                      "metric_early_late", "metric_10min", "metric_hourly"])
def test_smart_tolerance_reduces_interval_rows(sdk, scenario):
    times, values, period = ALL_SCENARIOS[scenario]()

    # Explicit tolerance 0 -> many rows.
    m0, d0 = new_md(sdk, _freq_for(period))
    sdk.write_time_value_pairs(m0, d0, times.copy(), values.copy(), period=period, time_units="ns")
    # write_data is reached with gap_tolerance forwarded; override to 0 via a direct call:
    rows_zero = _rows_with_explicit_tol(sdk, period, times, values, tol=0)

    # Smart default (gap_tolerance unset) -> far fewer rows.
    m1, d1 = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m1, d1, times.copy(), values.copy(), period=period, time_units="ns")
    rows_smart = n_interval_rows(sdk, m1, d1)

    assert rows_smart <= rows_zero
    assert rows_smart < max(2, rows_zero)  # genuine reduction for these jittery shapes


def _rows_with_explicit_tol(sdk, period, times, values, tol):
    m, d = new_md(sdk, _freq_for(period))
    sdk.write_data(m, d, times.copy(), values.copy(), period_ns=period, time_0=int(times[0]),
                   raw_time_type=TS, raw_value_type=1, encoded_value_type=3,
                   interval_index_mode="fast", gap_tolerance=tol, merge_blocks=False)
    return n_interval_rows(sdk, m, d)


# --------------------------------------------------------------------------- #
# 4. Encoding choices match expectations end-to-end (via stored block headers).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario,expected", [
    ("regular_waveform", (GAP, NONE)),
    ("metric_1hz", (GAP, NONE)),
    ("waveform_msg_jitter", (GAP, ZSTD)),
    ("waveform_float_time", (GAP, ZSTD)),
    ("metric_1_to_8s", (GAP, ZSTD)),
    ("metric_aperiodic", (TS, ZSTD)),
])
def test_encoding_choice_tvp(sdk, scenario, expected):
    times, values, period = ALL_SCENARIOS[scenario]()
    m, d = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")
        headers, _, _ = read_back(sdk, m, d, times, period)
    assert encodings(headers) == {expected}


def test_bounded_random_midrate_prefers_gap_via_measurement(sdk):
    # ~50% random sub-period deviations at 250 Hz: a compressed gap array is
    # smaller than a compressed timestamp array even though almost every gap
    # duration is distinct. The observed-spacing gate (median delta << 200ms =>
    # waveform, timestamp encoding not considered) settles this directly; for
    # slower signals the measured comparison still decides.
    times, values, period = sc_bounded_random_50pct()
    m, d = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")
        headers, rt, rv = read_back(sdk, m, d, times, period)
    assert encodings(headers) == {(GAP, ZSTD)}
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, values)


def test_segments_never_uses_timestamp_encoding(sdk):
    # Segment data is a gap array; even aperiodic data should stay gap-encoded.
    times, values, period = sc_metric_aperiodic()
    m, d = new_md(sdk, _freq_for(period))
    segs, starts = to_segments(times, values, period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seg, st in zip(segs, starts):
            sdk.write_segment(m, d, seg, st, period=period, time_units="ns")
        headers, rt, rv = read_back(sdk, m, d, times, period)
    assert all(int(h.t_encoded_type) == GAP for h in headers)
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, values)


# --------------------------------------------------------------------------- #
# 5. The `continuous` flag forces a single interval regardless of gaps.
# --------------------------------------------------------------------------- #
def test_continuous_time_value_pairs(sdk):
    # Large internal gaps that would otherwise split, even under the smart default.
    period = SEC
    times = np.array([0, 1, 2, 1000, 1001, 50000, 50001], dtype=np.int64) * SEC
    values = np.arange(times.size, dtype=np.int64)
    m, d = new_md(sdk, 1.0)
    sdk.write_time_value_pairs(m, d, times, values, period=period, time_units="ns", continuous=True)
    assert n_interval_rows(sdk, m, d) == 1
    _, rt, rv = read_back(sdk, m, d, times, period)
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, values)


def test_continuous_segments(sdk):
    period = SEC
    m, d = new_md(sdk, 1.0)
    segs = [np.arange(10, dtype=np.int64), np.arange(10, 20, dtype=np.int64)]
    starts = [0, 100000 * SEC]  # huge gap between the two segments
    sdk.write_segments(m, d, segs, starts, period=period, time_units="ns", continuous=True)
    assert n_interval_rows(sdk, m, d) == 1


def test_continuous_buffer(sdk):
    period = SEC
    m, d = new_md(sdk, 1.0)
    with sdk.write_buffer(continuous=True):
        sdk.write_segment(m, d, np.arange(10, dtype=np.int64), 0, period=period, time_units="ns")
        sdk.write_segment(m, d, np.arange(10, 20, dtype=np.int64), 100000 * SEC, period=period, time_units="ns")
    assert n_interval_rows(sdk, m, d) == 1


def test_non_continuous_splits_huge_gap(sdk):
    period = SEC
    times = np.array([0, 1, 2, 100000, 100001], dtype=np.int64) * SEC  # 100000s >> 10*period
    values = np.arange(times.size, dtype=np.int64)
    m, d = new_md(sdk, 1.0)
    sdk.write_time_value_pairs(m, d, times, values, period=period, time_units="ns")
    assert n_interval_rows(sdk, m, d) == 2  # the genuine outage still splits


# --------------------------------------------------------------------------- #
# 6. Explicit gap_tolerance=0 preserves the old behaviour (every gap splits).
# --------------------------------------------------------------------------- #
def test_explicit_zero_tolerance_records_every_gap(sdk):
    period = SEC
    times = np.array([0, 1, 2, 5, 6, 9], dtype=np.int64) * SEC  # gaps at 2->5 and 6->9
    values = np.arange(times.size, dtype=np.int64)
    rows = _rows_with_explicit_tol(sdk, period, times, values, tol=0)
    assert rows == 3


def test_buffered_write_keeps_gap_tolerance_on_its_measure_device_stream(sdk):
    """A buffer is only a batching mechanism: each write API owns its own
    interval setting, and buffering must preserve the immediate-write result."""
    period = SEC
    times = np.array([0, 1, 2, 5, 6, 9], dtype=np.int64) * SEC
    values = np.arange(times.size, dtype=np.int64)
    immediate_measure, immediate_device = new_md(sdk, 1.0)
    buffered_measure, buffered_device = new_md(sdk, 1.0)

    sdk.write_time_value_pairs(immediate_measure, immediate_device, times, values,
                               period=period, time_units="ns", gap_tolerance=0)
    with sdk.write_buffer():
        sdk.write_time_value_pairs(buffered_measure, buffered_device, times, values,
                                   period=period, time_units="ns", gap_tolerance=0)

    assert n_interval_rows(sdk, immediate_measure, immediate_device) == 3
    assert n_interval_rows(sdk, buffered_measure, buffered_device) == 3


def test_buffered_measure_device_rejects_conflicting_gap_tolerances(sdk):
    """Two buffered writes that will be merged for one measure/device cannot
    silently choose between incompatible caller-provided interval policies."""
    measure_id, device_id = new_md(sdk, 1.0)

    with pytest.raises(ValueError, match="gap_tolerance"):
        with sdk.write_buffer():
            sdk.write_time_value_pairs(measure_id, device_id, np.array([0], dtype=np.int64),
                                       np.array([1], dtype=np.int64), period=SEC,
                                       time_units="ns", gap_tolerance=0)
            sdk.write_time_value_pairs(measure_id, device_id, np.array([10 * SEC], dtype=np.int64),
                                       np.array([2], dtype=np.int64), period=SEC,
                                       time_units="ns", gap_tolerance=10 * SEC)


# --------------------------------------------------------------------------- #
# 7. Buffered and non-buffered produce identical results (encoding + intervals).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", list(ALL_SCENARIOS))
def test_buffered_matches_non_buffered(sdk, scenario):
    times, values, period = ALL_SCENARIOS[scenario]()

    m0, d0 = new_md(sdk, _freq_for(period))
    m1, d1 = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m0, d0, times.copy(), values.copy(), period=period, time_units="ns")
        with sdk.write_buffer():
            sdk.write_time_value_pairs(m1, d1, times.copy(), values.copy(), period=period, time_units="ns")
        h0, t0, v0 = read_back(sdk, m0, d0, times, period)
        h1, t1, v1 = read_back(sdk, m1, d1, times, period)

    np.testing.assert_array_equal(t0, t1)
    np.testing.assert_array_equal(v0, v1)
    assert encodings(h0) == encodings(h1)
    assert n_interval_rows(sdk, m0, d0) == n_interval_rows(sdk, m1, d1)


# --------------------------------------------------------------------------- #
# 8. Re-encoding (condense) preserves the smart time compression.
#    encode_blocks_from_multiple_segments must carry t_compression through, so a
#    block written as gap+zstd stays gap+zstd after reencode_dataset rather than
#    silently falling back to the instance-default (uncompressed) time data.
#    The transfer re-encode path uses the same function, so it is covered too.
# --------------------------------------------------------------------------- #
def test_reencode_preserves_zstd_time_compression(sdk):
    times, values, period = sc_waveform_float_time()  # structured deviations -> (GAP, ZSTD)
    m, d = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")

        # Precondition: the freshly written blocks are zstd-compressed gap arrays.
        headers_before, _, _ = read_back(sdk, m, d, times, period)
        assert encodings(headers_before) == {(GAP, ZSTD)}

        # Condense / re-encode the whole dataset.
        reencode_dataset(sdk, values_per_block=131072, blocks_per_file=2048, interval_gap_tolerance_nano=0)

        # The re-encoded blocks must keep the same encoding AND compression, and
        # the data must still read back exactly.
        headers_after, rt, rv = read_back(sdk, m, d, times, period)

    assert encodings(headers_after) == {(GAP, ZSTD)}, "reencode dropped the zstd time compression"
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, values)


# --------------------------------------------------------------------------- #
# 9. Block merging: write_time_value_pairs and write_segments now run the
#    small-write block merge by default (merge_blocks=True), so sequential
#    small writes build nice full linear blocks instead of fragmenting, and
#    the SQLite "merge" interval mode consolidates interval rows across writes.
# --------------------------------------------------------------------------- #
import sqlite3
from pathlib import Path

_COUNT_INTERVAL_ROWS = "SELECT COUNT(*) FROM interval_index WHERE measure_id = ? AND device_id = ?"


def _raw_interval_row_count(sdk, m, d):
    """Count physical interval_index rows (get_interval_array unions on read, so
    it can't see fragmentation)."""
    if sdk.metadata_connection_type == "sqlite":
        conn = sqlite3.connect(str(Path(sdk.dataset_location) / 'meta' / 'index.db'))
        try:
            return conn.execute(_COUNT_INTERVAL_ROWS, (m, d)).fetchone()[0]
        finally:
            conn.close()
    with sdk.sql_handler.maria_db_connection() as (conn, cursor):
        cursor.execute(_COUNT_INTERVAL_ROWS, (m, d))
        return cursor.fetchone()[0]


def test_sequential_tvp_writes_merge_into_one_block(sdk):
    period = SEC
    m, d = new_md(sdk, 1.0)
    n_chunk, n_chunks = 100, 5
    base = 1_600_000_000 * SEC
    for k in range(n_chunks):
        t = base + np.arange(k * n_chunk, (k + 1) * n_chunk, dtype=np.int64) * period
        v = np.arange(k * n_chunk, (k + 1) * n_chunk, dtype=np.int64)
        sdk.write_time_value_pairs(m, d, t, v, period=period, time_units="ns")

    all_t = base + np.arange(n_chunk * n_chunks, dtype=np.int64) * period
    headers, rt, rv = read_back(sdk, m, d, all_t, period)
    assert len(headers) == 1, "sequential small writes should merge into a single block"
    np.testing.assert_array_equal(rt, all_t)
    np.testing.assert_array_equal(rv, np.arange(n_chunk * n_chunks, dtype=np.int64))

    # And the interval index consolidated to a single physical row.
    assert _raw_interval_row_count(sdk, m, d) == 1


def test_merge_blocks_false_disables_merging(sdk):
    period = SEC
    m, d = new_md(sdk, 1.0)
    n_chunk, n_chunks = 100, 5
    base = 1_600_000_000 * SEC
    for k in range(n_chunks):
        t = base + np.arange(k * n_chunk, (k + 1) * n_chunk, dtype=np.int64) * period
        v = np.arange(k * n_chunk, (k + 1) * n_chunk, dtype=np.int64)
        sdk.write_time_value_pairs(m, d, t, v, period=period, time_units="ns", merge_blocks=False)

    all_t = base + np.arange(n_chunk * n_chunks, dtype=np.int64) * period
    headers, rt, rv = read_back(sdk, m, d, all_t, period)
    assert len(headers) == n_chunks, "merge_blocks=False must leave each write in its own block"
    np.testing.assert_array_equal(rt, all_t)


def test_sequential_write_segment_calls_merge_into_one_block(sdk):
    period = SEC
    m, d = new_md(sdk, 1.0)
    n_chunk, n_chunks = 100, 5
    base_s = 1_600_000_000
    for k in range(n_chunks):
        seg = np.arange(k * n_chunk, (k + 1) * n_chunk, dtype=np.int64)
        sdk.write_segment(m, d, seg, (base_s + k * n_chunk) * SEC, period=period, time_units="ns")

    all_t = base_s * SEC + np.arange(n_chunk * n_chunks, dtype=np.int64) * period
    headers, rt, rv = read_back(sdk, m, d, all_t, period)
    assert len(headers) == 1, "sequential write_segment calls should merge into a single block"
    np.testing.assert_array_equal(rt, all_t)
    np.testing.assert_array_equal(rv, np.arange(n_chunk * n_chunks, dtype=np.int64))
    assert _raw_interval_row_count(sdk, m, d) == 1


def test_merge_across_encoding_transition(sdk):
    """A regular chunk (auto -> raw gap array) followed by an aperiodic chunk
    (auto -> zstd timestamp array) must still merge: the encoding is re-chosen
    from the merged data instead of refusing on the encoding mismatch."""
    period = SEC
    m, d = new_md(sdk, 1.0)
    t1 = 1_600_000_000 * SEC + np.arange(500, dtype=np.int64) * period
    v1 = np.arange(500, dtype=np.int64)
    sdk.write_time_value_pairs(m, d, t1.copy(), v1.copy(), period=period, time_units="ns")

    deltas = np.maximum(1, np.round(_rng().exponential(1.0, size=400) * 1e9)).astype(np.int64)
    t2 = t1[-1] + np.cumsum(deltas)
    v2 = np.arange(500, 900, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, t2.copy(), v2.copy(), period=period, time_units="ns")

    all_t = np.concatenate((t1, t2))
    all_v = np.concatenate((v1, v2))
    headers, rt, rv = read_back(sdk, m, d, all_t, period)
    assert len(headers) == 1, "auto-encoded writes should merge across encoding transitions"
    np.testing.assert_array_equal(rt, all_t)
    np.testing.assert_array_equal(rv, all_v)


def test_buffer_flush_merges_with_existing_block(sdk):
    period = SEC
    m, d = new_md(sdk, 1.0)
    base = 1_600_000_000 * SEC
    t1 = base + np.arange(200, dtype=np.int64) * period
    v1 = np.arange(200, dtype=np.int64)
    sdk.write_time_value_pairs(m, d, t1.copy(), v1.copy(), period=period, time_units="ns")

    t2 = base + np.arange(200, 400, dtype=np.int64) * period
    v2 = np.arange(200, 400, dtype=np.int64)
    with sdk.write_buffer():
        sdk.write_time_value_pairs(m, d, t2.copy(), v2.copy(), period=period, time_units="ns")

    all_t = base + np.arange(400, dtype=np.int64) * period
    headers, rt, rv = read_back(sdk, m, d, all_t, period)
    assert len(headers) == 1, "a small buffered flush should merge with the existing block"
    np.testing.assert_array_equal(rt, all_t)
    np.testing.assert_array_equal(rv, np.arange(400, dtype=np.int64))


def test_fast_random_deviation_never_uses_timestamp_encoding(sdk):
    """A 250 Hz signal where every delta carries a random sub-period offset looks
    100% 'deviant', but its observed spacing is waveform-rate: the deviations are
    jitter/float noise, so the encoding must stay a compressed gap array."""
    period = SEC // 250
    n = 20000
    deltas = period + _rng().integers(1, period, size=n - 1)
    times = np.empty(n, dtype=np.int64)
    times[0] = 1_600_000_000 * SEC
    times[1:] = times[0] + np.cumsum(deltas)
    values = np.arange(n, dtype=np.int64)
    m, d = new_md(sdk, 250.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")
        headers, rt, rv = read_back(sdk, m, d, times, period)
    assert encodings(headers) == {(GAP, ZSTD)}
    np.testing.assert_array_equal(rt, times)


def test_small_scattered_slow_write_is_one_interval(sdk):
    """A small write of slow, scattered values: with so little data the whole
    call reads as continuous (the observed median spacing IS the scatter), so it
    lands as a single interval row instead of one row per value."""
    period = SEC
    times = np.array([0, 500 * SEC, 1200 * SEC], dtype=np.int64)
    values = np.arange(3, dtype=np.int64)
    m, d = new_md(sdk, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")
    assert n_interval_rows(sdk, m, d) == 1


def test_cluster_plus_outlier_splits_at_true_gap(sdk):
    """With enough data to see the cluster of typical spacings, only a value far
    outside it counts as a true gap."""
    period = 2 * SEC
    t_cluster = 1_600_000_000 * SEC + np.arange(50, dtype=np.int64) * period
    t_outlier = t_cluster[-1] + 10 * 86400 * SEC  # 10 days later
    times = np.concatenate((t_cluster, [t_outlier]))
    values = np.arange(times.size, dtype=np.int64)
    m, d = new_md(sdk, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m, d, times.copy(), values.copy(), period=period, time_units="ns")
    assert n_interval_rows(sdk, m, d) == 2


def test_one_value_at_a_time_intervals_evolve(sdk):
    """Aperiodic values written one call at a time: each small write merges into
    the growing block, the observed spacing is re-derived from the merged data,
    and the interval index evolves toward a handful of coarse rows - with no
    state beyond the block itself."""
    n = 150
    deltas = np.maximum(1, np.round(_rng().exponential(2.0, size=n - 1) * 1e9)).astype(np.int64)
    times = np.empty(n, dtype=np.int64)
    times[0] = 1_600_000_000 * SEC
    times[1:] = times[0] + np.cumsum(deltas)
    values = np.arange(n, dtype=np.int64)

    m, d = new_md(sdk, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            sdk.write_time_value_pairs(m, d, times[i:i + 1].copy(), values[i:i + 1].copy(),
                                       period=SEC, time_units="ns")

    row_count = _raw_interval_row_count(sdk, m, d)
    assert row_count <= 5, f"expected the index to evolve to a few coarse rows, got {row_count}"

    _, rt, rv = read_back(sdk, m, d, times, SEC)
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, values)


def test_merge_override_new_values_win_gap_path(sdk):
    """Overlapping writes through the gap-array merge path (write_segment):
    the newest write's values replace the old block's on shared timestamps."""
    period = SEC
    m, d = new_md(sdk, 1.0)
    base = 1_600_000_000 * SEC
    sdk.write_segment(m, d, np.arange(10, dtype=np.int64), base, period=period, time_units="ns")
    sdk.write_segment(m, d, np.full(4, 99, dtype=np.int64), base + 3 * SEC, period=period, time_units="ns")

    all_t = base + np.arange(10, dtype=np.int64) * period
    headers, rt, rv = read_back(sdk, m, d, all_t, period)
    assert len(headers) == 1
    np.testing.assert_array_equal(rt, all_t)
    np.testing.assert_array_equal(rv, np.array([0, 1, 2, 99, 99, 99, 99, 7, 8, 9], dtype=np.int64))


def test_merge_override_new_values_win_timestamp_path(sdk):
    """Overlapping writes through the timestamp-array merge path
    (write_time_value_pairs): the newest write's values win."""
    period = SEC
    m, d = new_md(sdk, 1.0)
    base = 1_600_000_000 * SEC
    times = base + np.arange(10, dtype=np.int64) * period
    sdk.write_time_value_pairs(m, d, times.copy(), np.arange(10, dtype=np.int64), period=period, time_units="ns")
    sdk.write_time_value_pairs(m, d, times[3:7].copy(), np.full(4, 99, dtype=np.int64),
                               period=period, time_units="ns")

    headers, rt, rv = read_back(sdk, m, d, times, period)
    assert len(headers) == 1
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, np.array([0, 1, 2, 99, 99, 99, 99, 7, 8, 9], dtype=np.int64))


def test_buffered_duplicate_pushes_newest_wins(sdk):
    """Duplicate timestamps pushed to the same buffer: the most recent push wins,
    matching the block-merge rule."""
    period = SEC
    m, d = new_md(sdk, 1.0)
    base = 1_600_000_000 * SEC
    times = base + np.arange(10, dtype=np.int64) * period
    with sdk.write_buffer():
        sdk.write_time_value_pairs(m, d, times.copy(), np.arange(10, dtype=np.int64),
                                   period=period, time_units="ns")
        sdk.write_time_value_pairs(m, d, times.copy(), np.arange(10, dtype=np.int64) + 100,
                                   period=period, time_units="ns")
    _, rt, rv = read_back(sdk, m, d, times, period)
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, np.arange(10, dtype=np.int64) + 100)


def test_interval_output_stable_across_batching(sdk):
    """The number of interval rows must come from predictable, data-driven rules:
    writing the same data in one call or in many chunks may only consolidate
    further (rows never multiply because of batching)."""
    times, values, period = sc_metric_aperiodic(n=600)

    m1, d1 = new_md(sdk, _freq_for(period))
    m2, d2 = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(m1, d1, times.copy(), values.copy(), period=period, time_units="ns")
        for ct, cv in zip(np.array_split(times, 12), np.array_split(values, 12)):
            sdk.write_time_value_pairs(m2, d2, ct.copy(), cv.copy(), period=period, time_units="ns")

    rows_single = _raw_interval_row_count(sdk, m1, d1)
    rows_chunked = _raw_interval_row_count(sdk, m2, d2)
    assert rows_single <= 3, f"aperiodic single write should stay sparse, got {rows_single}"
    assert rows_chunked <= rows_single, \
        f"batching multiplied interval rows: {rows_chunked} chunked vs {rows_single} single"

    for m, d in ((m1, d1), (m2, d2)):
        _, rt, rv = read_back(sdk, m, d, times, period)
        np.testing.assert_array_equal(rt, times)
        np.testing.assert_array_equal(rv, values)


def test_high_interval_density_logs_warning(sdk, caplog):
    """A write whose auto-chosen tolerance still leaves ~1 interval row per two
    values reports it. An explicitly requested tolerance never warns - recording
    every gap on purpose is the caller's decision."""
    import logging
    period = SEC
    m, d = new_md(sdk, 1.0)
    # Alternating 1s / 100s deltas: the median spacing is 1s, so the auto
    # tolerance is 10s and every 100s gap still splits -> ~50 rows for 100 values.
    deltas = np.tile(np.array([SEC, 100 * SEC], dtype=np.int64), 50)[:99]
    times = 1_600_000_000 * SEC + np.concatenate(([0], np.cumsum(deltas)))
    values = np.arange(100, dtype=np.int64)
    with caplog.at_level(logging.WARNING):
        sdk.write_data(m, d, times, values, period_ns=period, time_0=int(times[0]),
                       raw_time_type=TS, raw_value_type=1, encoded_value_type=3,
                       interval_index_mode="fast", merge_blocks=False)
    assert any("interval rows" in rec.message for rec in caplog.records)

    caplog.clear()
    m2, d2 = new_md(sdk, 1.0)
    with caplog.at_level(logging.WARNING):
        sdk.write_data(m2, d2, times, values, period_ns=period, time_0=int(times[0]),
                       raw_time_type=TS, raw_value_type=1, encoded_value_type=3,
                       interval_index_mode="fast", gap_tolerance=0, merge_blocks=False)
    assert not any("interval rows" in rec.message for rec in caplog.records)


def test_aperiodic_interval_rows_stay_sparse_across_writes(sdk):
    """Aperiodic data written in many small calls should leave only a handful of
    coarse interval rows (a general sense of where data exists), not one row per
    write or per gap."""
    times, values, period = sc_metric_aperiodic(n=1000, mean_s=2.0)
    m, d = new_md(sdk, _freq_for(period))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for chunk_t, chunk_v in zip(np.array_split(times, 10), np.array_split(values, 10)):
            sdk.write_time_value_pairs(m, d, chunk_t.copy(), chunk_v.copy(), period=period, time_units="ns")

    row_count = _raw_interval_row_count(sdk, m, d)
    # The smart tolerance is 10x the (modal) period; exponential arrivals with a
    # 2s mean rarely exceed that, so the index must stay far below one row per
    # write-call boundary or per gap.
    assert row_count < 50, f"expected a sparse interval index, got {row_count} rows"

    _, rt, rv = read_back(sdk, m, d, times, period)
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_array_equal(rv, values)
