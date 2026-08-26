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
"""Interval algebra behind ``AtriumSDK.get_event_intervals``.

Turning an event stream into ``from -> to`` state intervals is three separable
steps, and only the middle one needs the dataset:

1. **pair** the opening and closing events on the full stream
   (:func:`pair_from_to`),
2. **scope** the result to the ``within`` containers -- device/patient mappings or
   encounters -- which the SDK resolves because it must query them, and
3. **clip** the paired intervals to those containers, carrying the censoring
   flags across the cut (:func:`clip_intervals_to_containers`).

Steps 1 and 3 plus the container-normalizing :func:`union_windows` are pure
array/list math with no SDK state, so they live here and are directly testable
without a dataset. ``AtriumSDK`` reaches them through thin ``_pair_from_to`` /
``_clip_intervals_to_containers`` / ``_union_windows`` wrappers.

**Interval convention.** Every window and interval here is a half-open
``[start, end)`` pair of nanosecond integers, and every returned list is sorted
by start time.

**Censoring convention.** A boundary is never fabricated. When an interval's true
start or end is not observable -- the state was already open when the range or
container began, or never closed before it ended -- the interval is clipped to
that boundary and the corresponding ``start_censored`` / ``end_censored`` flag is
set. A caller that cannot tolerate a guessed boundary filters on the flags; the
timestamps themselves are always real boundaries of the *container*, never
invented event times.

"""
from __future__ import annotations

import numpy as np


def union_windows(windows):
    """Sort + merge overlapping/touching ``[start, end)`` windows (ns).

    Containers are built by unioning rows from several sources (many
    device_patient rows, several encounters), which can overlap or abut; the
    clip step assumes disjoint spans so that an interval cannot be emitted twice.
    Returns a new list of ``[start, end]`` lists; the input is untouched."""
    if not windows:
        return []
    ordered = sorted([int(s), int(e)] for s, e in windows)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def clip_spans_and_union(spans, start_n, end_n):
    """Clip ``(start, end)`` ns pairs to ``[start_n, end_n)`` and union the survivors.

    A ``None`` start means the row carries no span and is skipped; a ``None`` end
    means the span is still open and is clipped to ``end_n``. Spans that collapse to
    nothing after clipping are dropped. Returns disjoint ``[start, end]`` windows."""
    windows = []
    for s, e in spans:
        if s is None:
            continue
        s = max(int(s), start_n)
        e = min(int(e), end_n) if e is not None else end_n
        if s < e:
            windows.append([s, e])
    return union_windows(windows)


def pair_from_to(from_times, to_times, container_start, container_end):
    """Vectorized COLLAPSE pairing inside one span.

    ``from_times``/``to_times`` are sorted ns arrays already restricted to
    ``[container_start, container_end]``. Returns a list of
    ``(start, end, start_censored, end_censored)`` tuples, non-overlapping and
    sorted by start.

    Rule: a run of ``from``s until the next ``to`` is ONE interval (first-open ->
    first-close). A ``from`` with no following ``to`` in the span is
    right-censored to ``container_end``; a ``to`` before the first ``from`` (the
    span opened already inside the state) is left-censored from
    ``container_start``. No boundary is fabricated -- censored ends are clipped to
    the container, never invented.

    Implementation is fully vectorized: ``searchsorted`` maps each ``from`` to the
    first ``to`` strictly after it, and ``np.unique`` on that (monotonic) index
    collapses each run of ``from``s to its earliest member. No per-event loop.

    Precondition: ``from_times`` and ``to_times`` hold DISTINCT timestamps (no
    ``from`` coincident with a ``to``). The public path guarantees this -- storage
    dedups values at one timestamp to a single code (newest wins) -- so a ``from``
    and ``to`` can never share an exact ns. If this helper is called directly with
    coincident timestamps, ``side="right"`` treats the coincident ``to`` as not
    closing the ``from`` (a degenerate, non-reachable case)."""
    out = []
    n_from = from_times.shape[0]
    n_to = to_times.shape[0]

    # Leading left-censored interval: the span opened already inside the state
    # (a `to` occurs before any `from`). Only the first such `to` closes the
    # pre-existing open state; later stray `to`s while "out" are no-ops.
    if n_to > 0 and (n_from == 0 or to_times[0] < from_times[0]):
        out.append((int(container_start), int(to_times[0]), True, False))

    if n_from > 0:
        # For each `from`, index of the first `to` strictly greater than it;
        # == n_to when no `to` follows (right-censored). Monotonic because both
        # arrays are sorted.
        close_pos = np.searchsorted(to_times, from_times, side="right")
        # Collapse: froms sharing a close index are one run; the first (min time,
        # since from_times is sorted) opens the interval.
        uniq_close, first_idx = np.unique(close_pos, return_index=True)
        starts = from_times[first_idx]
        for close_idx, start in zip(uniq_close.tolist(), starts.tolist()):
            if close_idx == n_to:
                out.append((int(start), int(container_end), False, True))
            else:
                out.append((int(start), int(to_times[close_idx]), False, False))

    out.sort(key=lambda r: r[0])
    return out


def clip_intervals_to_containers(raw_intervals, windows):
    """Intersect paired ``(start, end, sc, ec)`` intervals (already censored
    relative to the whole query range) with the ``within`` container windows,
    carrying censoring flags. This is the same
    intersection math as ``intervals/intersection.list_intersection`` but done
    per (interval, window) so the censoring flags survive the clip:

      * an interval clipped at a container START becomes ``start_censored`` (the
        state was open before the container -- the far side is flagged, never a
        fabricated boundary);
      * clipped at a container END becomes ``end_censored``.

    A pair whose ``from`` lands in one window and ``to`` in the next is thereby
    split into a right-censored piece in the first window and a left-censored
    piece in the second, never crossing the gap between them."""
    out = []
    for (s, e, sc, ec) in raw_intervals:
        for w0, w1 in windows:
            cs = max(s, w0)
            ce = min(e, w1)
            if cs < ce:
                out.append((int(cs), int(ce),
                            bool(sc or s < w0),
                            bool(ec or e > w1)))
    out.sort(key=lambda r: r[0])
    return out


def collapse_event_intervals(times, codes, from_code, to_code,
                             range_start, range_end, windows):
    """Pair ``from``/``to`` on the FULL event stream over the query range (one
    vectorized :func:`pair_from_to`, censored to ``[range_start, range_end]``),
    then clip the result to the container windows via
    :func:`clip_intervals_to_containers`.

    Pairing on the full stream (rather than per-window slices) is what lets a
    container window that sits entirely inside an open state -- with no events of
    its own -- still be recognised as fully inside (both ends censored)."""
    from_times = times[codes == from_code]
    to_times = times[codes == to_code]
    raw = pair_from_to(from_times, to_times, range_start, range_end)
    return clip_intervals_to_containers(raw, windows)
