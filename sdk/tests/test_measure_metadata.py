# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
"""
Measure metadata tests cover signal_kind / value_type columns, schema upgrades,
read-time defaults, API plumbing, the get_data guard swap, and the numeric/string
write invariant.

SQLite only:
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_measure_metadata.py -q
"""
import shutil
import tempfile

import numpy as np
import pytest

from atriumdb import AtriumSDK

SEC = 1_000_000_000


@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_measure_metadata_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def times_for(n, start=1_600_000_000 * SEC, step=SEC):
    return start + np.arange(n, dtype=np.int64) * step


def _measure_columns(sdk):
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("PRAGMA table_info(measure)")
        return [row[1] for row in cursor.fetchall()]


def _set_raw_measure_columns(sdk, measure_id, signal_kind, value_type):
    """Force the raw columns (including NULL) directly, bypassing the SDK helpers,
    to simulate un-migrated / un-backfilled rows."""
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("UPDATE measure SET signal_kind = ?, value_type = ? WHERE id = ?",
                       (signal_kind, value_type, int(measure_id)))
        conn.commit()


def _drop_new_columns_legacy(sdk):
    """Recreate the (empty) measure table without signal_kind/value_type to
    simulate a legacy, column-less dataset. period_ns is kept."""
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("DROP TABLE measure")
        cursor.execute("""
            CREATE TABLE measure (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT NOT NULL,
            name TEXT NULL,
            freq_nhz INTEGER NOT NULL,
            period_ns INTEGER NULL,
            code TEXT NULL,
            unit TEXT NOT NULL,
            unit_label TEXT NULL,
            unit_code TEXT NULL,
            source_id INTEGER DEFAULT 1 NULL,
            UNIQUE (tag, freq_nhz, unit)
            )
        """)
        conn.commit()


# --------------------------------------------------------------------------- #
# Schema: fresh dataset + migration idempotency
# --------------------------------------------------------------------------- #
def test_fresh_dataset_has_new_columns(sdk):
    cols = _measure_columns(sdk)
    assert "signal_kind" in cols
    assert "value_type" in cols


def test_migration_adds_columns_and_is_idempotent(sdk):
    _drop_new_columns_legacy(sdk)
    cols = _measure_columns(sdk)
    assert "signal_kind" not in cols and "value_type" not in cols

    # First migration adds both columns.
    assert sdk.sql_handler.update_measure_schema() is True
    cols = _measure_columns(sdk)
    assert "signal_kind" in cols and "value_type" in cols

    # Re-running is a no-op.
    assert sdk.sql_handler.update_measure_schema() is False


# --------------------------------------------------------------------------- #
# Read-time defaults: NULL -> waveform / numeric
# --------------------------------------------------------------------------- #
def test_null_reads_as_waveform_numeric(sdk):
    m = sdk.insert_measure(measure_tag="hr", freq=1.0, freq_units="Hz", units="bpm")
    _set_raw_measure_columns(sdk, m, None, None)
    sdk._measures.pop(m, None)  # clear cache so a fresh DB read happens

    info = sdk.get_measure_info(m)
    assert info["signal_kind"] == "waveform"
    assert info["value_type"] == "numeric"


def test_explicit_values_round_trip_through_get_measure_info(sdk):
    m = sdk.insert_measure(measure_tag="nibp", freq=1.0, freq_units="Hz", units="mmHg",
                           signal_kind="sample", value_type="numeric")
    sdk._measures.pop(m, None)
    info = sdk.get_measure_info(m)
    assert info["signal_kind"] == "sample"
    assert info["value_type"] == "numeric"

    # get_all_measures reflects the same values.
    all_m = sdk.get_all_measures()
    assert all_m[m]["signal_kind"] == "sample"
    assert all_m[m]["value_type"] == "numeric"


def test_insert_measure_rejects_bad_enums(sdk):
    with pytest.raises(ValueError, match="signal_kind"):
        sdk.insert_measure(measure_tag="bad1", freq=1.0, freq_units="Hz", units="x",
                           signal_kind="continuous")
    with pytest.raises(ValueError, match="value_type"):
        sdk.insert_measure(measure_tag="bad2", freq=1.0, freq_units="Hz", units="x",
                           value_type="float")


# --------------------------------------------------------------------------- #
# String measures: first-write inference, backfill, and the get_data guard
# --------------------------------------------------------------------------- #
def test_string_first_write_infers_value_type_string(sdk):
    m = sdk.insert_measure(measure_tag="events", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev_events")
    sdk.write_time_value_pairs(m, d, times_for(3), np.array(["a", "b", "c"], dtype=object))

    sdk._measures.pop(m, None)
    assert sdk.get_measure_info(m)["value_type"] == "string"


def test_numeric_first_write_infers_value_type_numeric(sdk):
    m = sdk.insert_measure(measure_tag="num", freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag="dev_num")
    sdk.write_time_value_pairs(m, d, times_for(3), (np.arange(3)).astype(np.int64))

    sdk._measures.pop(m, None)
    assert sdk.get_measure_info(m)["value_type"] == "numeric"


def test_backfill_marks_string_measure(sdk):
    m = sdk.insert_measure(measure_tag="ev2", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev_ev2")
    sdk.write_time_value_pairs(m, d, times_for(3), np.array(["x", "y", "z"], dtype=object))

    # Simulate a dataset with a dictionary file and a NULL metadata column.
    _set_raw_measure_columns(sdk, m, None, None)
    sdk._measures.pop(m, None)

    # Read-time default already recovers 'string' via the dict-file fallback.
    assert sdk.get_measure_info(m)["value_type"] == "string"

    # The opportunistic backfill persists it into the column and is idempotent.
    sdk._backfill_string_value_types()
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("SELECT value_type FROM measure WHERE id = ?", (int(m),))
        assert cursor.fetchone()[0] == "string"
    sdk._backfill_string_value_types()  # re-run: no error, still 'string'


def test_get_data_guard_uses_value_type_column(sdk):
    m = sdk.insert_measure(measure_tag="ev3", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev_ev3")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))

    # The generic reader decodes strings when callers use its normal default.
    _, _, vals_from_get_data = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d, analog=True)
    assert list(vals_from_get_data) == ["a", "b", "c"]

    # get_string_data still works.
    _, vals = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert list(np.asarray(vals)) == ["a", "b", "c"]


def test_get_data_guard_falls_back_when_column_null(sdk):
    """Un-migrated string dataset (dict file present, value_type NULL) must still
    trip the guard via the fallback."""
    m = sdk.insert_measure(measure_tag="ev4", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev_ev4")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    _set_raw_measure_columns(sdk, m, None, None)  # column NULL, dict file remains
    sdk._measures.pop(m, None)

    _, _, values = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d, analog=True)
    assert list(values) == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Numeric/string write invariant
# --------------------------------------------------------------------------- #
def test_numeric_then_string_write_rejected(sdk):
    m = sdk.insert_measure(measure_tag="mix1", freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag="dev_mix1")
    t1 = times_for(3)
    sdk.write_time_value_pairs(m, d, t1, (np.arange(3)).astype(np.int64))
    t2 = times_for(3, start=int(t1[-1]) + SEC)
    with pytest.raises(ValueError, match="numeric"):
        sdk.write_time_value_pairs(m, d, t2, np.array(["a", "b", "c"], dtype=object))


def test_string_then_numeric_write_rejected(sdk):
    m = sdk.insert_measure(measure_tag="mix2", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev_mix2")
    t1 = times_for(3)
    sdk.write_time_value_pairs(m, d, t1, np.array(["a", "b", "c"], dtype=object))
    t2 = times_for(3, start=int(t1[-1]) + SEC)
    with pytest.raises(ValueError, match="string"):
        sdk.write_time_value_pairs(m, d, t2, (np.arange(3) + 10).astype(np.int64))


def test_explicit_value_type_enforced_before_any_write(sdk):
    """An explicitly-declared string measure rejects a numeric first write even
    before any data exists."""
    m = sdk.insert_measure(measure_tag="declared_str", freq=1.0, freq_units="Hz",
                           units="string", value_type="string")
    d = sdk.insert_device(device_tag="dev_declared")
    with pytest.raises(ValueError, match="string"):
        sdk.write_time_value_pairs(m, d, times_for(3), (np.arange(3)).astype(np.int64))
