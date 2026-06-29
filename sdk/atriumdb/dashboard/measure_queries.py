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

"""Dashboard-layer helpers for measure-level coverage statistics.

This module provides ``query_measure_total_hours``, which aggregates
``interval_index`` rows across all devices and returns the total wall-clock
hours of data stored per measure.

``interval_index`` holds one row per ``(measure_id, device_id)`` pair. Rows
within a pair are non-overlapping by the SDK's write invariant, so a plain
``SUM(end_time_n - start_time_n) GROUP BY measure_id`` gives an accurate
device-additive total without any double-counting within a single stream.

Only runs in direct-DB mode (``metadata_connection_type`` of ``"sqlite"``,
``"mysql"``, or ``"mariadb"``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atriumdb import AtriumSDK

logger = logging.getLogger(__name__)

_NS_PER_HOUR = 3_600_000_000_000

_MEASURE_TOTAL_HOURS_SQL = """
    SELECT
        m.id                                                        AS measure_id,
        m.measure_tag,
        m.freq_nhz,
        m.units,
        COUNT(DISTINCT ii.device_id)                                AS num_devices,
        SUM(ii.end_time_n - ii.start_time_n)                       AS total_ns
    FROM interval_index ii
    JOIN measures m ON m.id = ii.measure_id
    GROUP BY ii.measure_id
    ORDER BY total_ns DESC
"""

_MEASURE_TOTAL_HOURS_KEYS = (
    "measure_id",
    "measure_tag",
    "freq_nhz",
    "units",
    "num_devices",
    "total_ns",
)


def query_measure_total_hours(sdk: "AtriumSDK") -> list[dict]:
    """Return total wall-clock hours of coverage per measure across all devices.

    Queries ``interval_index`` (the SDK-maintained continuous-coverage table)
    and collapses every ``(measure_id, device_id)`` pair into a single
    measure-level total by grouping exclusively on ``measure_id``.

    ``interval_index`` must be populated (the default). If the dataset was
    ingested with ``interval_index_mode="disable"`` this function returns an
    empty list. See :func:`query_measure_total_hours_from_blocks` for a
    fallback that reads ``block_index`` instead.

    :param sdk: AtriumSDK instance in direct-DB mode.
    :return: List of dicts, one per measure that has at least one interval,
        ordered by ``total_ns`` descending::

            {
                "measure_id":  int,
                "measure_tag": str | None,
                "freq_nhz":    int,
                "units":       str | None,
                "num_devices": int,   # distinct devices that recorded this measure
                "total_ns":    int,   # total nanoseconds of coverage
                "total_hours": float, # total_ns / 3_600_000_000_000
            }
    """
    with sdk.sql_handler.connection(begin=False) as (conn, cursor):
        cursor.execute(_MEASURE_TOTAL_HOURS_SQL)
        rows = cursor.fetchall()

    result = []
    for row in rows:
        entry = dict(zip(_MEASURE_TOTAL_HOURS_KEYS, row))
        entry["total_hours"] = (entry["total_ns"] or 0) / _NS_PER_HOUR
        result.append(entry)

    logger.debug("query_measure_total_hours: returned %d measures", len(result))
    return result


# ---------------------------------------------------------------------------
# Fallback: block_index
# ---------------------------------------------------------------------------

_BLOCK_TOTAL_HOURS_SQL = """
    SELECT
        m.id                                                        AS measure_id,
        m.measure_tag,
        m.measure_name,
        m.freq_nhz,
        m.units,
        COUNT(DISTINCT bi.device_id)                                AS num_devices,
        COUNT(*)                                                    AS num_blocks,
        SUM(bi.num_values)                                          AS total_samples,
        SUM(bi.end_time_n - bi.start_time_n)                       AS total_ns
    FROM block_index bi
    JOIN measures m ON m.id = bi.measure_id
    GROUP BY bi.measure_id
    ORDER BY total_ns DESC
"""

_BLOCK_TOTAL_HOURS_KEYS = (
    "measure_id",
    "measure_tag",
    "measure_name",
    "freq_nhz",
    "units",
    "num_devices",
    "num_blocks",
    "total_samples",
    "total_ns",
)


def query_measure_total_hours_from_blocks(sdk: "AtriumSDK") -> list[dict]:
    """Fallback variant that reads ``block_index`` instead of ``interval_index``.

    Use this when ``interval_index`` is disabled or known to be stale. The
    result shape is the same as :func:`query_measure_total_hours` with two
    extra keys: ``num_blocks`` (replaces ``num_intervals``) and
    ``total_samples`` (sum of ``block_index.num_values``).

    Note: ``total_hours`` counts wall-clock span of each block including any
    intra-block gaps. ``total_samples`` counts only the actual recorded
    data points. They diverge when signals have internal gaps encoded in the
    TSC file.

    :param sdk: AtriumSDK instance in direct-DB mode.
    :return: List of dicts, one per measure, ordered by ``total_ns`` descending::

            {
                "measure_id":    int,
                "measure_tag":   str | None,
                "measure_name":  str | None,
                "freq_nhz":      int,
                "units":         str | None,
                "num_devices":   int,
                "num_blocks":    int,
                "total_samples": int,
                "total_ns":      int,
                "total_hours":   float,
            }
    """
    with sdk.sql_handler.connection(begin=False) as (conn, cursor):
        cursor.execute(_BLOCK_TOTAL_HOURS_SQL)
        rows = cursor.fetchall()

    result = []
    for row in rows:
        entry = dict(zip(_BLOCK_TOTAL_HOURS_KEYS, row))
        entry["total_hours"] = (entry["total_ns"] or 0) / _NS_PER_HOUR
        result.append(entry)

    logger.debug(
        "query_measure_total_hours_from_blocks: returned %d measures", len(result)
    )
    return result
