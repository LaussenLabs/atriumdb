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
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from bisect import bisect_right

from atriumdb.windowing.window import Window
from atriumdb.string_dictionary import UNKNOWN_STRING_CODE
from atriumdb.measure_kinds import (
    SIGNAL_KIND_WAVEFORM, SIGNAL_KIND_SAMPLE, SIGNAL_KIND_EVENT, SIGNAL_KIND_STATE,
    DEFAULT_SIGNAL_KIND, is_string_value_type, measure_kind_of)

# 1 second nominal raster period, used when an aperiodic measure has no usable
# grid period.
ONE_SECOND_NS = 1_000_000_000

# --- fill rule names ------------------------------------------------------- #
# The internal waveform-numeric path: the window IS the measure's stored sample
# grid, NaN where there is no sample. Never user-selectable.
FILL_GRID = "grid"
# The value most recently in effect holds until the next reading (sample/state).
FILL_CARRY_FORWARD = "carry_forward"
# Only the cell a reading falls in is populated; every other cell is unknown.
FILL_SPARSE = "sparse"
# Occupancy of a grid cell: was there any event (0/1), or how many.
FILL_PRESENCE = "presence"
FILL_COUNT = "count"
FILL_AGGREGATE_LAST = "aggregate:last"
FILL_AGGREGATE_MEAN = "aggregate:mean"
FILL_AGGREGATE_MIN = "aggregate:min"
FILL_AGGREGATE_MAX = "aggregate:max"

# Rules that report OCCUPANCY of a cell rather than a value observed in it.
# They are always numeric 0/1 (or a count), absence is a meaningful 0 rather
# than an unknown, and they are the reason a string ``event`` measure's window
# holds floats and cannot be decoded back to strings.
OCCUPANCY_FILL_RULES = (FILL_PRESENCE, FILL_COUNT)

# Rules that reduce several readings in one cell with arithmetic, so they are
# meaningless for dictionary codes ("the mean of ASYSTOLE and V-TACH").
NUMERIC_AGGREGATE_FILL_RULES = (FILL_AGGREGATE_MEAN, FILL_AGGREGATE_MIN, FILL_AGGREGATE_MAX)

# Per-signal_kind default fill rules.
_DEFAULT_FILL_FOR_KIND = {
    SIGNAL_KIND_WAVEFORM: FILL_GRID,      # existing NaN grid -- numeric path, untouched
    SIGNAL_KIND_SAMPLE: FILL_CARRY_FORWARD,
    SIGNAL_KIND_STATE: FILL_CARRY_FORWARD,  # carry-forward with left-censoring
    SIGNAL_KIND_EVENT: FILL_PRESENCE,
}

# Which fill rules each signal_kind accepts. "grid" is the waveform-numeric
# path and is never selected for aperiodic kinds.
_RULES_FOR_KIND = {
    SIGNAL_KIND_WAVEFORM: (FILL_GRID,),
    SIGNAL_KIND_SAMPLE: (FILL_CARRY_FORWARD, FILL_SPARSE, FILL_AGGREGATE_LAST,
                         FILL_AGGREGATE_MEAN, FILL_AGGREGATE_MIN, FILL_AGGREGATE_MAX),
    SIGNAL_KIND_STATE: (FILL_CARRY_FORWARD,),
    SIGNAL_KIND_EVENT: OCCUPANCY_FILL_RULES,
}

# Every fill rule name the aperiodic renderer understands, across all kinds.
# "grid" is the internal waveform-numeric path and is never user-selectable, so
# it is deliberately excluded from the user-facing valid set.
SUPPORTED_FILL_RULES = tuple(sorted(
    {rule for kind, rules in _RULES_FOR_KIND.items()
     if kind != SIGNAL_KIND_WAVEFORM for rule in rules}))


def validate_fill_rule_name(rule, param_name="aperiodic_fill"):
    """Reject a fill rule that is not a supported rule name *at all*.

    Kind-compatibility is a separate, later check (an incompatible-but-valid
    global default falls back to the kind default; an incompatible per-measure
    override raises). This guard exists so that outright garbage or a typo such
    as ``"carry-forward"`` can never be silently swallowed."""
    if rule is None:
        return
    if not isinstance(rule, str) or rule not in SUPPORTED_FILL_RULES:
        raise ValueError(
            f"{param_name}: '{rule}' is not a supported fill rule. "
            f"Valid rules are: {', '.join(SUPPORTED_FILL_RULES)}.")


def default_fill_for_kind(signal_kind):
    """The default fill rule for a signal_kind."""
    return _DEFAULT_FILL_FOR_KIND.get(signal_kind, FILL_CARRY_FORWARD)


def resolve_fill_rule(signal_kind, value_type, override=None, global_default=None):
    """Resolve the effective fill rule for one measure.

    Precedence: a per-measure ``override``
    (``fill_overrides[measure_id]``) wins and is validated strictly (an
    incompatible rule for the kind raises). Otherwise a ``global_default``
    (``get_iterator(aperiodic_fill=...)``) is applied when it is *compatible*
    with the kind, else the per-kind default is used. ``waveform`` always
    resolves to the untouched ``grid`` path.

    String value_type additionally forbids numeric reductions
    (``aggregate:mean|min|max``) -- only ``aggregate:last`` makes sense for
    codes.
    """
    # Reject unsupported rule NAMES before anything else, so a typo can never be
    # silently swallowed by the waveform short-circuit or the compatibility
    # fallback below.
    validate_fill_rule_name(global_default, param_name="aperiodic_fill")
    validate_fill_rule_name(override, param_name="fill_overrides")

    if signal_kind == SIGNAL_KIND_WAVEFORM:
        if override is not None:
            raise ValueError(
                f"Fill rule '{override}' cannot be applied to a 'waveform' measure; "
                f"waveform measures always use the NaN sample grid. Remove this "
                f"measure from fill_overrides.")
        return FILL_GRID

    allowed = _RULES_FOR_KIND.get(signal_kind, ())

    def _reject_string_numeric_agg(rule):
        if is_string_value_type(value_type) and rule in NUMERIC_AGGREGATE_FILL_RULES:
            raise ValueError(
                f"Fill rule '{rule}' is numeric and cannot apply to a string measure; "
                f"only 'aggregate:last' (or 'carry_forward'/'sparse') is valid for string codes.")

    if override is not None:
        if override not in allowed:
            raise ValueError(
                f"Fill rule '{override}' is not valid for a '{signal_kind}' measure. "
                f"Allowed: {allowed}.")
        _reject_string_numeric_agg(override)
        return override

    if global_default is not None and global_default in allowed:
        try:
            _reject_string_numeric_agg(global_default)
            return global_default
        except ValueError:
            # A global default that is incompatible with this particular
            # (string) measure silently falls back to the per-kind default;
            # only an explicit per-measure override raises.
            pass

    return default_fill_for_kind(signal_kind)


def resolve_nominal_period_ns(measure, period_override=None):
    """Resolve a measure's nominal raster period.

    Order: ``get_iterator`` override -> ``measure['period_ns']`` for waveform ->
    1 s default for aperiodic kinds. The freq-derived ``period_ns`` carried on an
    aperiodic measure is *not* a meaningful raster period, so aperiodic kinds
    grid at 1 s unless explicitly overridden."""
    if period_override is not None:
        return int(period_override)
    signal_kind = measure.get("signal_kind") or DEFAULT_SIGNAL_KIND
    if signal_kind == SIGNAL_KIND_WAVEFORM:
        return int(measure["period_ns"])
    return ONE_SECOND_NS


def build_render_config(measures, window_duration_ns, window_slide_ns, aperiodic_fill=None,
                        fill_overrides=None, period_overrides=None):
    """Resolve the raster configuration shared by window-producing call sites.

    ``DatasetIterator`` and ``DatasetDefinition.filter`` must render a measure
    identically before they expose it to user code. Keeping the resolution and
    zero-cell checks here prevents one path from silently falling back to the
    legacy numeric grid.
    """
    fill_overrides = dict(fill_overrides) if fill_overrides else {}
    period_overrides = dict(period_overrides) if period_overrides else {}

    known_ids = {measure['id'] for measure in measures}
    id_hint = ", ".join(f"{measure['id']} ({measure['tag']})" for measure in measures)
    for param_name, overrides in (("fill_overrides", fill_overrides),
                                  ("period_overrides", period_overrides)):
        unknown = [key for key in overrides if key not in known_ids]
        if unknown:
            raise ValueError(
                f"{param_name} contains key(s) {unknown} that match no measure in this "
                f"definition. Keys must be measure IDs (integers), not measure tags. "
                f"Measures in this definition: {id_hint}.")

    config = {}
    for measure in measures:
        measure_id = measure['id']
        measure_name = f"Measure {measure_id} ('{measure['tag']}')"
        signal_kind, value_type = measure_kind_of(measure)
        period_override = period_overrides.get(measure_id)
        if period_override is not None and signal_kind == SIGNAL_KIND_WAVEFORM:
            raise ValueError(
                f"{measure_name} is a 'waveform' measure; period_overrides only applies to "
                f"aperiodic measures ('sample'/'state'/'event'), whose nominal raster period "
                f"is a rendering choice. A waveform is always sampled on its own stored "
                f"period ({measure['period_ns']} ns). Remove measure {measure_id} from "
                f"period_overrides.")

        period_ns = int(resolve_nominal_period_ns(measure, period_override=period_override))
        if period_ns > window_duration_ns:
            raise ValueError(
                f"{measure_name}: resolved nominal raster period {period_ns} ns is larger than "
                f"the window duration {window_duration_ns} ns, so a window would contain zero "
                f"grid cells. Increase window_duration or lower this measure's period via "
                f"period_overrides.")
        if period_ns > window_slide_ns:
            raise ValueError(
                f"{measure_name}: resolved nominal raster period {period_ns} ns is larger than "
                f"the window slide {window_slide_ns} ns, so the slide would advance zero grid "
                f"cells. Increase window_slide or lower this measure's period via "
                f"period_overrides.")

        fill_rule = resolve_fill_rule(
            signal_kind, value_type, override=fill_overrides.get(measure_id),
            global_default=aperiodic_fill)
        config[measure_id] = {
            'signal_kind': signal_kind,
            'value_type': value_type,
            'period_ns': period_ns,
            'fill_rule': fill_rule,
            'is_string': is_string_value_type(value_type),
        }
    return config


def _rasterize_grid(grid_times, period_ns, sample_times, sample_values, rule, is_string,
                    seed_time=None, seed_value=None):
    """Rasterize sparse readings onto a regular grid per one fill ``rule``.

    Returns ``(values, known)`` where ``known`` is a *separable* internal boolean
    array -- "is this cell a genuine, observed value" -- kept separable so a real
    per-signal ``known`` mask can be emitted additively later. The caller applies
    the sentinel; here we already fold the
    sentinel into ``values`` (NaN for float channels, the reserved
    ``UNKNOWN_STRING_CODE`` for string/int64 channels) using ``known``.

    ``grid_times`` is the evenly spaced grid (cell i spans
    ``[grid_times[i], grid_times[i] + period_ns)``). ``sample_times`` must be
    ascending (as returned by ``get_data(sort=True)``).

    ``seed_time`` / ``seed_value`` (carry-forward only) is the single most-recent
    reading that occurred *before* this batch grid, within the caller's bounded
    lookback horizon (see ``CARRY_FORWARD_LOOKBACK_NS``). It makes carry-forward
    deterministic and batch-size independent: cells before the first in-batch
    reading carry the seed value instead of being left-censored. It is always
    earlier than ``grid_times[0]``.
    """
    n = int(grid_times.shape[0])

    # event presence/count is always numeric 0/1 (or a count); absence is a
    # meaningful 0, so every cell is "known" and there is no sentinel.
    if rule in OCCUPANCY_FILL_RULES:
        values = np.zeros(n, dtype=np.float64)
        known = np.ones(n, dtype=bool)
        if n == 0 or sample_times.size == 0:
            return values, known
        grid_start = grid_times[0]
        cell = ((np.asarray(sample_times, dtype=np.int64) - grid_start) // period_ns).astype(np.int64)
        m = (cell >= 0) & (cell < n)
        cell = cell[m]
        counts = np.bincount(cell, minlength=n)[:n]
        values = (counts > 0).astype(np.float64) if rule == FILL_PRESENCE else counts.astype(np.float64)
        return values, known

    # Non-event kinds: unknown cells carry a sentinel.
    if is_string:
        values = np.full(n, UNKNOWN_STRING_CODE, dtype=np.int64)
    else:
        values = np.full(n, np.nan, dtype=np.float64)
    known = np.zeros(n, dtype=bool)

    # Normalize the sample arrays to the channel's target dtype (int64 codes for
    # string channels, float64 for numeric).
    target_dtype = np.int64 if is_string else np.float64
    st = np.asarray(sample_times, dtype=np.int64)
    sv = np.asarray(sample_values).astype(target_dtype, copy=False)

    # Carry-forward seed: prepend the pre-batch reading so
    # cells before the first in-batch reading carry the value already in effect.
    # The seed is always earlier than grid_times[0], so after this merge every
    # grid cell has a prior reading and carry-forward is batch-size independent.
    if rule == FILL_CARRY_FORWARD and seed_time is not None:
        st = np.concatenate((np.asarray([seed_time], dtype=np.int64), st))
        sv = np.concatenate((np.asarray([seed_value], dtype=target_dtype), sv))

    if n == 0 or st.size == 0:
        return values, known

    grid_start = grid_times[0]

    if rule == FILL_CARRY_FORWARD:
        # Each cell reports the value in effect DURING the cell -- i.e. the most
        # recent reading strictly before the cell's END, where cell i spans
        # [grid_times[i], grid_times[i] + period_ns).
        #
        # Deliberately NOT `searchsorted(st, grid_times, side="right")`, which would
        # take the most recent reading at or before the cell's START. That silently
        # DELAYS every mid-cell reading by one cell and, at the end of a definition
        # range, DROPS it outright: a lone reading landing in the final cell has no
        # later cell to surface in, so the window comes back all-sentinel with
        # actual_count == 0 despite a genuine observation inside the range. It would
        # also leave carry_forward using a different cell-attribution convention from
        # `sparse`/`aggregate:*` (which bucket a reading into the cell it falls in).
        # Attributing a reading to the cell it occurs in fixes both: no reading inside
        # the range can be invisible, and all aperiodic rules now agree on which cell
        # a reading belongs to. Readings exactly on a cell boundary are unaffected
        # (side="left" on the cell end keeps them in the cell they open), so the
        # boundary-aligned behaviour every existing test pins is byte-identical.
        #
        # Cells before the first reading (and with no seed) are still left-censored ->
        # sentinel (known stays False). This is the state/sample carry-forward
        # machinery.
        idx = np.searchsorted(st, grid_times + period_ns, side="left") - 1
        valid = idx >= 0
        sel = np.where(valid, idx, 0)
        picked = sv[sel]
        if is_string:
            values = np.where(valid, picked, UNKNOWN_STRING_CODE)
        else:
            values = np.where(valid, picked, np.nan)
        known = valid
        return values, known

    # sparse / aggregate: bucket readings by the cell they fall in.
    cell = ((st - grid_start) // period_ns).astype(np.int64)
    m = (cell >= 0) & (cell < n)
    c = cell[m]
    v = sv[m]
    if c.size == 0:
        return values, known

    if rule in (FILL_SPARSE, FILL_AGGREGATE_LAST):
        # Later (more recent) readings in the same cell overwrite earlier ones.
        if is_string:
            values[c] = v.astype(np.int64)
        else:
            values[c] = v.astype(np.float64)
        known[c] = True
        return values, known

    # Numeric reductions (guarded to numeric measures by resolve_fill_rule).
    vf = v.astype(np.float64)
    counts = np.bincount(c, minlength=n)[:n]
    nz = counts > 0
    if rule == FILL_AGGREGATE_MEAN:
        sums = np.bincount(c, weights=vf, minlength=n)[:n]
        values[nz] = sums[nz] / counts[nz]
    elif rule == FILL_AGGREGATE_MIN:
        acc = np.full(n, np.inf)
        np.minimum.at(acc, c, vf)
        values[nz] = acc[nz]
    elif rule == FILL_AGGREGATE_MAX:
        acc = np.full(n, -np.inf)
        np.maximum.at(acc, c, vf)
        values[nz] = acc[nz]
    known[nz] = True
    return values, known


# Carry-forward seed lookback tuning. The seed read walks BACKWARDS from the
# batch start in geometrically growing chunks and stops at the first chunk that
# contains a reading, so the decoded volume is proportional to the local data
# density near the batch boundary rather than to the whole elapsed range.
_SEED_LOOKBACK_BASE_CELLS = 4     # first chunk = 4 grid cells
_SEED_LOOKBACK_GROWTH = 8         # each empty chunk widens the search 8x

# How far BEFORE the definition's range start carry-forward may look for the
# reading already in effect at the range start. A `sample` cell holds "the most
# recent prior reading"; the unknown sentinel is reserved for genuinely-unknown
# cells.
#
# Why a horizon at all, and why this one:
#
# * It must be a function of the DEFINITION's range start alone -- never of the
#   batch, of ``num_windows_prefetch`` or of a ``cached_windows_per_source``
#   split -- or batching would change rendered values. Subtracting a constant
#   from the definition range start preserves that property exactly.
# * It must be finite. An unbounded "scan back to the beginning of time" makes
#   the first window of every cohort pay for the whole history of the stream.
# * 24 h is longer than the charting interval of every routinely-recorded slow
#   clinical measure (NIBP q5-15 min, labs q6-24 h, vent settings per change),
#   so the flagship "N minutes either side of every event" recipe renders its
#   pre-anchor half from the value actually in effect instead of NaN. It is also
#   the point past which "the most recent prior reading" stops meaning "the
#   value in effect now": carrying a >1-day-old observation forward would be
#   fabricating currency rather than recovering it.
# * The cost of the horizon is not proportional to it. The seed read walks
#   backwards in geometrically growing chunks and stops at the first non-empty
#   one, so a horizon this wide costs ~6 index probes that decode nothing when
#   there is genuinely no prior reading, and one small decode when there is.
CARRY_FORWARD_LOOKBACK_NS = 24 * 60 * 60 * 1_000_000_000


def _fetch_carry_forward_seed(sdk, measure_id, device_id, query_patient_id,
                              seed_floor, data_start_time, period_ns, analog):
    """Most recent reading in ``[seed_floor, data_start_time)``, read in bounded chunks.

    Returns ``(seed_time, seed_value)`` or ``(None, None)``.

    A single ``get_data(seed_floor, data_start_time)`` would decode the entire
    elapsed range on *every* batch -- O(N) per batch, O(N^2) overall -- while
    keeping only the last element. Instead scan backwards: chunks are contiguous
    and ordered most-recent-first, so the first non-empty chunk necessarily holds
    the latest reading before ``data_start_time``. The answer is therefore
    identical to the full-range read, but empty chunks decode nothing and the
    single non-empty chunk is bounded by the local reading density.
    """
    span = int(data_start_time) - int(seed_floor)
    if span <= 0:
        return None, None

    chunk = max(int(period_ns) * _SEED_LOOKBACK_BASE_CELLS, 1)
    chunk_end = int(data_start_time)
    while chunk_end > seed_floor:
        chunk_start = max(int(seed_floor), chunk_end - chunk)
        _, s_times, s_values = sdk.get_data(
            measure_id, chunk_start, chunk_end, device_id=device_id,
            patient_id=query_patient_id, analog=analog)
        s_times = np.asarray(s_times)
        if s_times.size > 0:
            return int(np.asarray(s_times, dtype=np.int64)[-1]), np.asarray(s_values)[-1]
        chunk_end = chunk_start
        chunk *= _SEED_LOOKBACK_GROWTH
    return None, None


class _BatchGrid:
    """The geometry of one measure's batch array: the evenly spaced timestamps
    the batch covers, plus which slice of them is the part actually inside the
    definition's range.

    Both the legacy waveform path and the aperiodic fill path need exactly this
    arithmetic, and they computed it independently -- twenty duplicated lines
    where an edit to one copy silently diverged the two paths. It is pure
    arithmetic on ints, so hoisting it changes no value.

    ``times`` spans ``batch_num_windows`` windows' worth of cells starting at
    ``batch_start_time``. That is deliberately LONGER than the range whenever a
    trailing partial window overhangs the range end, so ``data_slice`` (from
    ``start_index``, of length ``expected_num_values``) marks the in-range part:
    the caller fills that and leaves the overhang at whatever sentinel it
    pre-filled the array with.
    """

    __slots__ = ("times", "window_size", "slide_size", "start_index",
                 "expected_num_values", "data_start_time", "data_end_time")

    def __init__(self, period_ns, window_duration_ns, window_slide_ns, batch_start_time,
                 batch_end_time, batch_num_windows, range_start_time, range_end_time):
        self.window_size = int(window_duration_ns // period_ns)
        self.slide_size = int(window_slide_ns // period_ns)
        batch_size = self.window_size + (batch_num_windows - 1) * self.slide_size
        quantized_end_time = batch_start_time + (batch_size * period_ns)
        self.times = np.arange(batch_start_time, quantized_end_time, period_ns)

        # If partial windows are allowed, we need to make room for an extra full window,
        # but then only partially populate it. So find just the region where we actually want data
        self.data_start_time = max(range_start_time, batch_start_time)
        self.data_end_time = min(range_end_time, batch_end_time)

        self.start_index = np.searchsorted(self.times, self.data_start_time, side='left')

        self.expected_num_values = int(round(
            (self.data_end_time - self.data_start_time) / period_ns))
        room = self.times.size - self.start_index
        if self.expected_num_values > room:
            self.data_end_time = self.data_start_time + int(round(room * period_ns))
            self.expected_num_values = int(round(
                (self.data_end_time - self.data_start_time) / period_ns))

    @property
    def data_slice(self):
        """The ``slice`` of the batch arrays that lies inside the definition range."""
        return slice(self.start_index, self.start_index + self.expected_num_values)

    def sliding_windows(self, value_array):
        """Cut ``(times, values)`` into the batch's overlapping windows."""
        windowed_times = sliding_window_view(self.times, self.window_size)
        windowed_values = sliding_window_view(value_array, self.window_size)
        return windowed_times[::self.slide_size], windowed_values[::self.slide_size]


def get_threshold_labels(sliced_labels, label_threshold=0.5):
    # Calculate the percentage of 1s for each label in each window
    percentages = np.mean(sliced_labels, axis=-1)
    # Apply threshold
    return (percentages > label_threshold).astype(int)


def find_closest_measurement(time, measurements):
    """
    Find the measurement with the time value closest, but less than or equal to the given time,
    directly using bisect on the list of tuples.

    :param time: The time (epoch timestamp) to find the closest measurement for.
    :param measurements: A list of tuples containing the measurement value, units, and epoch timestamp.
    :return: The tuple from the measurements list with the closest time less than or equal to the given time.
    """
    # Use bisect_right with a key function that extracts the timestamp part of the tuple
    idx = bisect_right(measurements, time, key=lambda x: x[5])

    if idx > 0:
        return measurements[idx - 1]
    else:
        return None


def get_signal_dictionary(sdk, device_id, query_patient_id, window_duration_ns, window_slide_ns, measures,
                          batch_start_time, batch_end_time, batch_num_windows, range_start_time, range_end_time,
                          render_config=None, definition_range_start_time=None,
                          carry_forward_lookback_ns=None):
    """Build the per-measure sliding windows for one batch.

    ``render_config`` maps ``measure_id -> {signal_kind, value_type,
    period_ns, fill_rule, is_string}``. When it is ``None`` (older iterators /
    the definition-filter path) or a measure resolves to the untouched
    waveform-numeric case, the exact legacy NaN-grid path runs -- this keeps the
    numeric windowing path byte-for-byte identical. Aperiodic / string measures
    take the per-kind fill path instead.

    ``carry_forward_lookback_ns`` bounds how far before the definition's range
    start a carry_forward measure may look for the reading already in effect;
    ``None`` uses :data:`CARRY_FORWARD_LOOKBACK_NS`. It is deliberately a
    function of the definition range only, so no batching knob can change a
    rendered value.
    """
    if carry_forward_lookback_ns is None:
        carry_forward_lookback_ns = CARRY_FORWARD_LOOKBACK_NS
    # Reset and populate the batch data signal dictionary
    source_batch_data_dictionary = {}
    for i, measure in enumerate(measures):
        measure_id = measure['id']
        cfg = render_config.get(measure_id) if render_config else None

        # ---- Legacy waveform-numeric path (unchanged, byte-for-byte) -------- #
        if cfg is None or cfg['fill_rule'] == FILL_GRID:
            period_ns = measure['period_ns'] if cfg is None else cfg['period_ns']

            grid = _BatchGrid(period_ns, window_duration_ns, window_slide_ns, batch_start_time,
                              batch_end_time, batch_num_windows, range_start_time, range_end_time)
            measure_filled_value_array = np.full(grid.times.shape, np.nan)

            nan_filled_out = measure_filled_value_array[grid.data_slice]

            if grid.expected_num_values > 0:
                sdk.get_data(
                    measure_id, grid.data_start_time, grid.data_end_time, device_id=device_id,
                    patient_id=query_patient_id, return_nan_filled=nan_filled_out)

            # Create the windows and slide them.
            sliced_windowed_measure_times, sliced_windowed_measure_values = \
                grid.sliding_windows(measure_filled_value_array)

            # Store the measure's time and value arrays in the batch data dictionary
            source_batch_data_dictionary[measure_id] = \
                (sliced_windowed_measure_times, sliced_windowed_measure_values, grid.window_size)
            continue

        # ---- aperiodic / string fill path ---------------------------------- #
        period_ns = cfg['period_ns']
        rule = cfg['fill_rule']
        is_string = cfg['is_string']

        grid = _BatchGrid(period_ns, window_duration_ns, window_slide_ns, batch_start_time,
                          batch_end_time, batch_num_windows, range_start_time, range_end_time)
        measure_filled_time_array = grid.times

        # Whole-batch value array pre-filled with the sentinel.
        #
        # Event presence/count cells are a meaningful 0 ("no event occurred")
        # ONLY inside the definition range. The batch array is longer than the
        # range whenever a trailing partial window overhangs the range end, so
        # those overhanging cells must NOT be a fabricated 0 -- that would be
        # indistinguishable from a genuine "no alarm" and reported by
        # actual_count as fully covered. They are NaN (the same unknown sentinel
        # every other float channel uses out of range); the in-range slice below
        # is overwritten wholesale with the real 0/1 (or count) grid, so genuine
        # absence inside the range is still a hard 0.
        if is_string and rule not in OCCUPANCY_FILL_RULES:
            measure_filled_value_array = np.full(
                measure_filled_time_array.shape, UNKNOWN_STRING_CODE, dtype=np.int64)
        else:
            measure_filled_value_array = np.full(
                measure_filled_time_array.shape, np.nan, dtype=np.float64)

        data_start_time = grid.data_start_time
        data_end_time = grid.data_end_time
        expected_num_values = grid.expected_num_values

        if expected_num_values > 0:
            grid_slice = measure_filled_time_array[grid.data_slice]
            # String measures read raw int64 codes (analog=False); numeric
            # aperiodic measures read analog-scaled values. Neither uses the
            # numeric return_nan_filled path.
            _, r_times, r_values = sdk.get_data(
                measure_id, data_start_time, data_end_time, device_id=device_id,
                patient_id=query_patient_id, analog=not is_string)

            # Carry-forward seed: makes carry-forward deterministic and
            # independent of the batch size (num_windows_prefetch). The per-batch
            # read only covers [data_start_time, data_end_time); a reading that
            # precedes this batch would otherwise be invisible and a genuinely
            # KNOWN cell would be emitted as the unknown sentinel. So, for a
            # carry_forward measure, fetch the single most-recent reading in
            # [seed_floor, data_start_time) and seed the grid with it. The floor is
            # anchored on the DEFINITION's range start (not the per-batch sub-range
            # start, which the batcher moves forward every batch) so a value set
            # early in the range still carries into a later batch.
            #
            # The anchor is then set back by CARRY_FORWARD_LOOKBACK_NS. Flooring
            # the seed AT the definition range start made the same wall-clock
            # window render differently depending on where the cohort's region
            # happened to begin: a reading 70 s before a region start was in the
            # database, inside one unbroken availability interval, and still came
            # back NaN with actual_count == 0 for minutes -- so every "N minutes
            # either side of an event" window rendered its whole pre-anchor half as
            # missing data for slow measures. The horizon is still a pure function
            # of the definition range start, so batching independence is unchanged;
            # see CARRY_FORWARD_LOOKBACK_NS for why it is finite and why 24 h.
            # The lookback itself is performed by _fetch_carry_forward_seed in
            # geometrically growing chunks walking BACKWARDS from data_start_time,
            # so peak RAM per call and total decode work stay bounded by the local
            # reading density instead of by the elapsed range. (A single
            # [seed_floor, data_start_time) read was O(N) per batch and O(N^2)
            # overall, despite an earlier comment here claiming otherwise.)
            # presence/count/sparse/aggregate touch only a reading's own cell and
            # are already batch-independent -- no seed.
            seed_anchor = range_start_time if definition_range_start_time is None \
                else definition_range_start_time
            # max(0, ...) keeps the floor a valid timestamp for datasets whose
            # epoch starts at (or near) 0, which every test fixture here does.
            seed_floor = max(0, int(seed_anchor) - int(carry_forward_lookback_ns))
            seed_time = seed_value = None
            if rule == FILL_CARRY_FORWARD and data_start_time > seed_floor:
                seed_time, seed_value = _fetch_carry_forward_seed(
                    sdk, measure_id, device_id, query_patient_id, seed_floor,
                    data_start_time, period_ns, analog=not is_string)

            values, _known = _rasterize_grid(
                grid_slice, period_ns, np.asarray(r_times, dtype=np.int64), r_values, rule, is_string,
                seed_time=seed_time, seed_value=seed_value)
            measure_filled_value_array[grid.data_slice] = values

        sliced_windowed_measure_times, sliced_windowed_measure_values = \
            grid.sliding_windows(measure_filled_value_array)

        source_batch_data_dictionary[measure_id] = \
            (sliced_windowed_measure_times, sliced_windowed_measure_values, grid.window_size)
    return source_batch_data_dictionary


def get_label_data(device_id, query_patient_id, batch_start_time, batch_end_time, batch_time_array, sdk,
                   row_size, slide_size, label_sets, label_threshold, label_exact_match=False):
    sliced_labels, threshold_labels = None, None
    # If labels exist, calculate them
    if len(label_sets) > 0:
        # Preallocate label matrix
        label_matrix = np.zeros((len(label_sets), len(batch_time_array)), dtype=np.int8)

        # Populate label matrix
        for idx, label_set_id in enumerate(label_sets):
            sdk.get_label_time_series(
                label_name_id=label_set_id,
                device_id=device_id if device_id else None,
                patient_id=query_patient_id if query_patient_id else None,
                start_time=batch_start_time,
                end_time=batch_end_time,
                timestamp_array=batch_time_array,
                out=label_matrix[idx],
                include_descendants=not label_exact_match,
            )

        # Create label windows
        windowed_label_views = sliding_window_view(
            label_matrix, (len(label_sets), row_size), axis=None)
        sliced_labels = windowed_label_views[0][::slide_size]

        threshold_labels = get_threshold_labels(sliced_labels, label_threshold=label_threshold)
    return sliced_labels, threshold_labels


def get_label_dictionary(sdk, device_id, query_patient_id, source_batch_start_time, source_batch_end_time, label_sets,
                         label_threshold, range_num_windows, row_period_ns, row_size, slide_size, label_exact_match=False):
    range_size = row_size + (range_num_windows - 1) * slide_size
    quantized_end_time = source_batch_start_time + (range_size * row_period_ns)
    source_time_array = np.arange(source_batch_start_time, quantized_end_time, row_period_ns)
    sliced_labels, threshold_labels = get_label_data(
        device_id, query_patient_id, source_batch_start_time, source_batch_end_time, source_time_array,
        sdk, row_size, slide_size, label_sets, label_threshold, label_exact_match=label_exact_match
    )
    return sliced_labels, threshold_labels

def _get_patient_info_from_cache(patient_id, window_start_time, patient_info_cache, patient_history_cache):
    window_patient_info = patient_info_cache.get(patient_id, {})
    for field, history_timeseries in patient_history_cache.get(patient_id, {}).items():
        best_match = find_closest_measurement(window_start_time, history_timeseries)
        if best_match is None:
            continue

        _, _, _, value, units, time = best_match
        window_patient_info[field] = {
            'value': value,
            'units': units,
            'time': time,
        }
    return window_patient_info

def get_window_list(device_id, patient_id, validated_measure_list, source_batch_data_dictionary,
                    batch_start_time, num_windows, window_duration_ns, window_slide_ns, threshold_labels, sliced_labels,
                    patient_history_cache, patient_history_fields, patient_info_cache):
    batch_window_list = []
    window_start_time = batch_start_time
    for window_i in range(num_windows):
        signal_dictionary = {}
        for measure in validated_measure_list:
            measure_id = measure['id']
            measure_tag = measure['tag']
            measure_freq_nhz = measure['freq_nhz']
            measure_freq_hz = float(measure_freq_nhz / (10 ** 9))
            measure_units = measure['units']

            window_times = source_batch_data_dictionary[measure_id][0][window_i]

            window_values = source_batch_data_dictionary[measure_id][1][window_i]
            measure_expected_count = source_batch_data_dictionary[measure_id][2]

            # actual_count = number of genuinely known cells. Float channels use
            # NaN as the unknown sentinel (waveform / numeric aperiodic); int64
            # string-code channels use the reserved UNKNOWN_STRING_CODE. The
            # float branch is left exactly as before (no numeric-path change).
            if np.issubdtype(window_values.dtype, np.floating):
                actual_count = np.sum(~np.isnan(window_values))
            else:
                actual_count = np.sum(window_values != UNKNOWN_STRING_CODE)

            signal_dictionary[(measure_tag, measure_freq_hz, measure_units)] = \
                {
                    'times': np.copy(window_times),
                    'values': np.copy(window_values),
                    'expected_count': measure_expected_count,
                    'actual_count': actual_count,
                    'measure_id': measure_id,
                }

        label_time_series = np.copy(sliced_labels[window_i]) if \
            sliced_labels is not None else None

        window_classification = threshold_labels[window_i] if \
            threshold_labels is not None else None

        window_patient_info = patient_info_cache.get(patient_id, {})
        if patient_history_fields:
            window_patient_info = _get_patient_info_from_cache(
                patient_id, window_start_time, patient_info_cache, patient_history_cache)

        result_window = Window(
            signals=signal_dictionary,
            start_time=int(window_start_time),
            end_time=int(window_start_time + window_duration_ns),
            device_id=device_id,
            patient_id=patient_id,
            label_time_series=label_time_series,
            label=window_classification,
            patient_info=window_patient_info
        )
        batch_window_list.append(result_window)
        window_start_time += window_slide_ns
    return batch_window_list


def _load_patient_cache(patient_id, patient_info_cache, patient_history_cache, sdk, patient_history_fields):
    if patient_id is None:
        return
    # Check if patient id is in the info cache.
    if patient_id not in patient_info_cache:
        patient_info_cache[patient_id] = sdk.get_patient_info(patient_id=patient_id)
        if patient_info_cache[patient_id] is None:
            patient_info_cache[patient_id] = {}
        else:
            # Delete Height And Weight From Static Info
            if 'height' in patient_info_cache[patient_id]:
                del patient_info_cache[patient_id]['height']

            if 'weight' in patient_info_cache[patient_id]:
                del patient_info_cache[patient_id]['weight']

    if patient_history_fields and patient_id not in patient_history_cache:
        patient_history_cache[patient_id] = {}
        for field in ['height', 'weight']:
            if field not in patient_history_fields:
                continue
            patient_history_cache[patient_id][field] = sdk.get_patient_history(
                patient_id=patient_id, field=field)
