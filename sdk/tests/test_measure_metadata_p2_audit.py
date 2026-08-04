# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
"""
INDEPENDENT ADVERSARIAL AUDIT of Phase 2 "measure metadata" (design section 19).

This file probes BEYOND the writer's own test_measure_metadata_p2.py. It tries to
break the migration, the read-time defaults, the numeric/string mix-rejection
invariant on EVERY write entry point, persistence, caching, detection fallbacks,
and do-no-harm on interval reads. Bugs are captured as xfail(strict=True) tests so
that fixing the source flips them to PASS without editing this file.

SQLite (the gate):
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_measure_metadata_p2_audit.py -q

MariaDB (optional; container atriumdb-mariadb up):
    ... add -e MARIA_DB_HOST=host.docker.internal
"""
import os
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from atriumdb import AtriumSDK

SEC = 1_000_000_000


# --------------------------------------------------------------------------- #
# SQLite fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_p2audit_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def reopen(sdk_obj, **kwargs):
    """Reopen the same location as a brand-new SDK object (fresh caches)."""
    loc = sdk_obj._loc
    return AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite", **kwargs)


def times_for(n, start=1_600_000_000 * SEC, step=SEC):
    return start + np.arange(n, dtype=np.int64) * step


def measure_columns(sdk_obj):
    with sdk_obj.sql_handler.connection() as (conn, cursor):
        cursor.execute("PRAGMA table_info(measure)")
        return [row[1] for row in cursor.fetchall()]


def set_raw_columns(sdk_obj, measure_id, signal_kind, value_type):
    with sdk_obj.sql_handler.connection() as (conn, cursor):
        cursor.execute("UPDATE measure SET signal_kind = ?, value_type = ? WHERE id = ?",
                       (signal_kind, value_type, int(measure_id)))
        conn.commit()


def read_raw_value_type(sdk_obj, measure_id):
    with sdk_obj.sql_handler.connection() as (conn, cursor):
        cursor.execute("SELECT value_type FROM measure WHERE id = ?", (int(measure_id),))
        return cursor.fetchone()[0]


def drop_p2_columns(sdk_obj):
    """Simulate a P0/P1 dataset that has period_ns but NOT the P2 columns."""
    with sdk_obj.sql_handler.connection() as (conn, cursor):
        cursor.execute("ALTER TABLE measure DROP COLUMN signal_kind")
        cursor.execute("ALTER TABLE measure DROP COLUMN value_type")
        conn.commit()


STR = lambda vals: np.array(vals, dtype=object)
NUM = lambda vals: np.array(vals, dtype=np.int64)

BASE = 1_600_000_000 * SEC


# =========================================================================== #
# 1. MIGRATION
# =========================================================================== #
def test_migration_idempotent_three_runs(sdk):
    drop_p2_columns(sdk)
    assert "signal_kind" not in measure_columns(sdk)
    assert sdk.sql_handler.update_measure_schema() is True   # adds both
    assert sdk.sql_handler.update_measure_schema() is False  # no-op
    assert sdk.sql_handler.update_measure_schema() is False  # still no-op
    cols = measure_columns(sdk)
    assert "signal_kind" in cols and "value_type" in cols


def test_migration_adds_only_missing_column(sdk):
    """period_ns present, one P2 column present, one missing -> add just the missing
    one (exercises the independent per-column guards, not an all-or-nothing ALTER)."""
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("ALTER TABLE measure DROP COLUMN value_type")  # keep signal_kind
        conn.commit()
    assert "signal_kind" in measure_columns(sdk)
    assert "value_type" not in measure_columns(sdk)
    assert sdk.sql_handler.update_measure_schema() is True
    cols = measure_columns(sdk)
    assert "signal_kind" in cols and "value_type" in cols


def test_reopen_without_auto_upgrade_raises_clear_error(sdk):
    """Opening a pre-P2 (column-less) dataset with the DEFAULT auto_upgrade=False and
    then reading a measure must NOT surface a raw 'no such column' sqlite error; it
    should raise a clear ValueError pointing at auto_upgrade=True. (The message is
    misattributed to 'period_ns' even when it is the P2 columns that are missing --
    a minor UX nit, documented here.)"""
    sdk.insert_measure(measure_tag="hr", freq=1.0, freq_units="Hz", units="bpm")
    drop_p2_columns(sdk)
    sdk.close()
    # The default-connect path eagerly primes get_all_measures in __init__, so the
    # clear ValueError surfaces at construction time (fail-fast), not on first read.
    with pytest.raises(ValueError, match="auto_upgrade"):
        AtriumSDK(dataset_location=sdk._loc, metadata_connection_type="sqlite")


def test_reopen_with_auto_upgrade_migrates_and_reads(sdk):
    """auto_upgrade=True on a column-less dataset runs the ALTERs and reads default
    correctly to waveform/numeric."""
    m = sdk.insert_measure(measure_tag="hr", freq=1.0, freq_units="Hz", units="bpm")
    drop_p2_columns(sdk)
    sdk.close()
    s2 = AtriumSDK(dataset_location=sdk._loc, metadata_connection_type="sqlite", auto_upgrade=True)
    try:
        assert "signal_kind" in measure_columns(s2)
        info = s2.get_measure_info(m)
        assert (info["signal_kind"], info["value_type"]) == ("waveform", "numeric")
    finally:
        s2.close()


def test_auto_upgrade_backfills_string_measure_on_connect(sdk):
    """A P1 string dataset (dict file present) that is missing the P2 columns should,
    on an auto_upgrade connect, migrate AND opportunistically backfill value_type."""
    m = sdk.insert_measure(measure_tag="ev", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev")
    sdk.write_time_value_pairs(m, d, times_for(3), STR(["a", "b", "c"]))
    drop_p2_columns(sdk)  # now looks like a P1-only dataset with a dict file
    sdk.close()
    s2 = AtriumSDK(dataset_location=sdk._loc, metadata_connection_type="sqlite", auto_upgrade=True)
    try:
        assert read_raw_value_type(s2, m) == "string"   # backfill persisted it
        assert s2.get_measure_info(m)["value_type"] == "string"
    finally:
        s2.close()


# =========================================================================== #
# 2. READ-TIME DEFAULTS
# =========================================================================== #
def test_null_defaults_everywhere(sdk):
    m = sdk.insert_measure(measure_tag="hr", freq=1.0, freq_units="Hz", units="bpm")
    set_raw_columns(sdk, m, None, None)
    sdk._measures.pop(m, None)

    info = sdk.get_measure_info(m)
    assert (info["signal_kind"], info["value_type"]) == ("waveform", "numeric")
    assert sdk.get_measure_kind(m) == ("waveform", "numeric")

    all_m = sdk.get_all_measures()
    assert (all_m[m]["signal_kind"], all_m[m]["value_type"]) == ("waveform", "numeric")

    # Raw select_all_measures rows still carry NULLs (defaults are applied above them).
    rows = {r[0]: r for r in sdk.sql_handler.select_all_measures()}
    assert rows[m][10] is None and rows[m][11] is None


def test_null_value_type_with_dict_file_reads_string(sdk):
    """value_type NULL but a P1 dictionary file exists -> string, through every surface."""
    m = sdk.insert_measure(measure_tag="ev", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev")
    sdk.write_time_value_pairs(m, d, times_for(3), STR(["a", "b", "c"]))
    set_raw_columns(sdk, m, None, None)  # column NULL, dict file remains on disk
    sdk._measures.pop(m, None)

    assert sdk.get_measure_info(m)["value_type"] == "string"
    assert sdk.get_measure_kind(m)[1] == "string"
    assert sdk.get_all_measures()[m]["value_type"] == "string"


def test_get_measure_kind_missing_measure_returns_none(sdk):
    assert sdk.get_measure_kind(999999) is None


# =========================================================================== #
# 3. MIX-REJECTION INVARIANT ON EVERY WRITE ENTRY POINT
# =========================================================================== #
def _numeric_measure(sdk, tag):
    m = sdk.insert_measure(measure_tag=tag, freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag=tag + "_dev")
    sdk.write_time_value_pairs(m, d, times_for(3, start=BASE), NUM([1, 2, 3]))
    return m, d


def _string_measure(sdk, tag):
    m = sdk.insert_measure(measure_tag=tag, freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag=tag + "_dev")
    sdk.write_time_value_pairs(m, d, times_for(3, start=BASE), STR(["a", "b", "c"]))
    return m, d


def test_mix_reject_write_time_value_pairs_both_directions(sdk):
    m, d = _numeric_measure(sdk, "tvp_num")
    with pytest.raises(ValueError):
        sdk.write_time_value_pairs(m, d, times_for(3, start=10_000 * SEC), STR(["x", "y", "z"]))
    m2, d2 = _string_measure(sdk, "tvp_str")
    with pytest.raises(ValueError):
        sdk.write_time_value_pairs(m2, d2, times_for(3, start=10_000 * SEC), NUM([1, 2, 3]))


def test_mix_reject_write_data_both_directions(sdk):
    from atriumdb.block_wrapper import T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
    m, d = _numeric_measure(sdk, "wd_num")
    t = times_for(3, start=10_000 * SEC)
    with pytest.raises(ValueError):
        sdk.write_data(m, d, t, STR(["x", "y", "z"]), freq_nhz=SEC, time_0=int(t[0]),
                       raw_time_type=T_TYPE_TIMESTAMP_ARRAY_INT64_NANO)
    m2, d2 = _string_measure(sdk, "wd_str")
    with pytest.raises(ValueError):
        sdk.write_data(m2, d2, t, NUM([1, 2, 3]), freq_nhz=SEC, time_0=int(t[0]),
                       raw_time_type=T_TYPE_TIMESTAMP_ARRAY_INT64_NANO)


def test_mix_reject_write_segments(sdk):
    m, d = _numeric_measure(sdk, "seg_num")
    with pytest.raises(ValueError):
        sdk.write_segments(m, d, [STR(["x", "y", "z"])], [10_000.0], freq=1.0, freq_units="Hz")
    m2, d2 = _string_measure(sdk, "seg_str")
    with pytest.raises(ValueError):
        sdk.write_segments(m2, d2, [NUM([1, 2, 3])], [10_000.0], freq=1.0, freq_units="Hz")


def test_mix_reject_write_buffer_flush(sdk):
    """The invariant is enforced at flush (write_data) time; a conflicting buffered
    write must still be rejected, not silently persisted."""
    m, d = _numeric_measure(sdk, "buf_num")
    with pytest.raises(ValueError):
        with sdk.write_buffer():
            sdk.write_segments(m, d, [STR(["x", "y", "z"])], [10_000.0], freq=1.0, freq_units="Hz")


def test_mix_reject_write_data_easy_numeric_measure_gets_string(sdk):
    """write_data_easy is numeric-only by design; a string array must fail, not corrupt."""
    m, d = _numeric_measure(sdk, "easy_num")
    with pytest.raises(ValueError):
        sdk.write_data_easy(m, d, times_for(3, start=10_000 * SEC), STR(["x", "y", "z"]),
                            freq=1, freq_units="Hz")


def test_second_numeric_write_not_falsely_rejected(sdk):
    """A SECOND numeric write to an established-numeric measure must still work."""
    m, d = _numeric_measure(sdk, "num_twice")
    sdk.write_time_value_pairs(m, d, times_for(3, start=BASE + 100 * SEC), NUM([4, 5, 6]))
    _, rt, rv = sdk.get_data(m, BASE - SEC, BASE + 200 * SEC, device_id=d)
    assert rt.size == 6


def test_second_string_write_not_falsely_rejected(sdk):
    m, d = _string_measure(sdk, "str_twice")
    sdk.write_time_value_pairs(m, d, times_for(3, start=BASE + 100 * SEC), STR(["d", "e", "f"]))
    _, vals = sdk.get_string_data(m, BASE - SEC, BASE + 200 * SEC, device_id=d)
    assert sorted(map(str, vals)) == ["a", "b", "c", "d", "e", "f"]


def test_explicit_numeric_measure_rejects_string_before_write(sdk):
    """Reverse of the writer's declared-string test: an explicitly numeric measure
    rejects a string first write before any data/dict file is created."""
    m = sdk.insert_measure(measure_tag="declared_num", freq=1.0, freq_units="Hz",
                           units="x", value_type="numeric")
    d = sdk.insert_device(device_tag="declared_num_dev")
    with pytest.raises(ValueError):
        sdk.write_time_value_pairs(m, d, times_for(3), STR(["a", "b", "c"]))
    from atriumdb.string_dictionary import MeasureStringDictionary
    assert not MeasureStringDictionary.exists(sdk._meta_dir, m)  # no dict file leaked


# =========================================================================== #
# 4. PERSISTENCE ACROSS A FRESH SDK OBJECT
# =========================================================================== #
def test_established_value_type_persists_across_reopen_numeric(sdk):
    m, d = _numeric_measure(sdk, "persist_num")
    assert read_raw_value_type(sdk, m) == "numeric"
    sdk.close()
    s2 = reopen(sdk)
    try:
        assert s2.get_measure_info(m)["value_type"] == "numeric"
        # And the invariant still holds against the reopened, cache-cold object.
        with pytest.raises(ValueError):
            s2.write_time_value_pairs(m, d, times_for(3, start=9_000 * SEC), STR(["a", "b"]))
    finally:
        s2.close()


def test_established_value_type_persists_across_reopen_string(sdk):
    m, d = _string_measure(sdk, "persist_str")
    assert read_raw_value_type(sdk, m) == "string"
    sdk.close()
    s2 = reopen(sdk)
    try:
        assert s2.get_measure_info(m)["value_type"] == "string"
        with pytest.raises(ValueError):
            s2.write_time_value_pairs(m, d, times_for(3, start=9_000 * SEC), NUM([1, 2]))
    finally:
        s2.close()


# =========================================================================== #
# 5. CACHING
# =========================================================================== #
def test_cache_not_stale_after_first_write_establishes_type(sdk):
    """insert_measure caches value_type='numeric' for a fresh measure. A first string
    write must invalidate that cache so a later get_measure_info reflects 'string'."""
    m = sdk.insert_measure(measure_tag="cache", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="cache_dev")
    assert sdk.get_measure_info(m)["value_type"] == "numeric"  # prime the cache
    sdk.write_time_value_pairs(m, d, times_for(3), STR(["a", "b", "c"]))
    # No manual cache pop here: the write path must have invalidated it.
    assert sdk.get_measure_info(m)["value_type"] == "string"
    assert sdk.get_measure_kind(m)[1] == "string"


# =========================================================================== #
# 6. DETECTION FALLBACK / INCONSISTENT STATE
# =========================================================================== #
def test_column_string_but_no_dict_file_get_data_still_guards(sdk):
    """Inconsistent forced state: value_type='string' column on a measure that only
    ever had numeric data and no dict file. The get_data guard trusts the column and
    rejects analog reads (documents that the column is authoritative over reality)."""
    m, d = _numeric_measure(sdk, "inconsistent")
    set_raw_columns(sdk, m, None, "string")
    sdk._measures.pop(m, None)
    with pytest.raises(ValueError, match="string measure"):
        sdk.get_data(m, 0, 10_000 * SEC, device_id=d, analog=True)


# =========================================================================== #
# 7. DO-NO-HARM
# =========================================================================== #
def test_get_interval_array_unaffected_numeric_and_string(sdk):
    mn, dn = _numeric_measure(sdk, "iv_num")
    ms, ds = _string_measure(sdk, "iv_str")
    iv_n = sdk.get_interval_array(mn, device_id=dn, gap_tolerance_nano=0)
    iv_s = sdk.get_interval_array(ms, device_id=ds, gap_tolerance_nano=0)
    assert iv_n.shape[1] == 2 and iv_n.size > 0
    assert iv_s.shape[1] == 2 and iv_s.size > 0


def test_transfer_measures_do_no_harm(sdk):
    """Do-no-harm (spec section 19.6): transferring measures between two datasets must
    not CHOKE on the new columns. P2 deliberately does NOT carry signal_kind/value_type
    (that is P6), so the destination measures default to waveform/numeric -- this test
    documents that current, intended behavior and that the transfer completes."""
    src = sdk
    m1 = src.insert_measure(measure_tag="plain", freq=1.0, freq_units="Hz", units="x")
    m2 = src.insert_measure(measure_tag="nibp", freq=1.0, freq_units="Hz", units="mmHg",
                            signal_kind="sample", value_type="numeric")
    dst_loc = tempfile.mkdtemp(prefix="atrium_p2audit_dst_")
    shutil.rmtree(dst_loc, ignore_errors=True)
    dst = AtriumSDK.create_dataset(dataset_location=dst_loc, database_type="sqlite")
    try:
        from atriumdb.transfer.adb.measures import transfer_measures
        transfer_measures(src, dst, measure_id_list=[m1, m2])
        dst_measures = {mm["tag"]: mm for mm in dst.get_all_measures().values()}
        assert "plain" in dst_measures and "nibp" in dst_measures
        # Columns are NOT carried in P2: destination defaults apply.
        assert dst_measures["nibp"]["value_type"] == "numeric"
        assert dst_measures["nibp"]["signal_kind"] == "waveform"
    except ImportError:
        pytest.skip("transfer_measures helper not importable in this build")
    finally:
        dst.close()
        shutil.rmtree(dst_loc, ignore_errors=True)


# =========================================================================== #
# 8. CONCURRENCY (best-effort)
# =========================================================================== #
def test_concurrent_first_writes_converge_to_one_type(sdk):
    """Two threads racing to establish value_type on a brand-new measure. Whatever
    wins, the persisted type must be consistent and reads must not corrupt."""
    m = sdk.insert_measure(measure_tag="race", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="race_dev")
    errors = []

    def w(start):
        try:
            sdk.write_time_value_pairs(m, d, times_for(3, start=start * SEC),
                                       STR([f"v{start}a", f"v{start}b", f"v{start}c"]))
        except Exception as e:  # a rejected racer is acceptable; a crash-y one is noted
            errors.append(repr(e))

    threads = [threading.Thread(target=w, args=(s,)) for s in (1000, 2000)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Regardless of race outcome the measure is string and reads back cleanly.
    assert read_raw_value_type(sdk, m) == "string"
    _, vals = sdk.get_string_data(m, 0, 5000 * SEC, device_id=d)
    assert all(isinstance(v, str) for v in np.asarray(vals))


# =========================================================================== #
# 9. REGRESSION: value_type-poisoning bug found by this audit, now FIXED.
#    value_type is established only AFTER a write commits, so a write that raises
#    cannot leave a poisoning value_type. (These were xfail; they now pass.)
# =========================================================================== #
def test_failed_write_data_easy_string_must_not_poison_measure(sdk):
    m = sdk.insert_measure(measure_tag="poison", freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag="poison_dev")
    with pytest.raises(ValueError):
        sdk.write_data_easy(m, d, times_for(3), STR(["a", "b", "c"]), freq=1, freq_units="Hz")
    # Desired behavior: the rejected write left no established type, so a numeric write works.
    sdk.write_time_value_pairs(m, d, times_for(3), NUM([1, 2, 3]))
    _, rt, rv = sdk.get_data(m, 0, 10 * SEC + int(times_for(3)[0]), device_id=d)
    assert rt.size == 3


def test_failed_write_data_conflicting_rawtype_must_not_poison_measure(sdk):
    from atriumdb.block_wrapper import T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, V_TYPE_DOUBLE
    m = sdk.insert_measure(measure_tag="poison2", freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag="poison2_dev")
    t = times_for(3)
    with pytest.raises(ValueError):
        sdk.write_data(m, d, t, STR(["a", "b", "c"]), freq_nhz=SEC, time_0=int(t[0]),
                       raw_time_type=T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, raw_value_type=V_TYPE_DOUBLE)
    sdk.write_time_value_pairs(m, d, t, NUM([1, 2, 3]))  # should work; currently rejected
    _, rt, rv = sdk.get_data(m, 0, int(t[-1]) + SEC, device_id=d)
    assert rt.size == 3


# =========================================================================== #
# 10. MARIADB (optional) -- the migration on a PRE-EXISTING column-less dataset,
#     which the writer did NOT directly exercise.
# =========================================================================== #
def _maria_params():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    host = os.getenv("MARIA_DB_HOST")
    if host is None:
        return None
    return {
        "host": host,
        "user": os.getenv("MARIA_DB_USER", "root"),
        "password": os.getenv("MARIA_DB_PASSWORD", "atriumdb"),
        "port": int(os.getenv("MARIA_DB_PORT", "3306")),
    }


@pytest.mark.parametrize("_", [0])
def test_maria_migration_on_preexisting_columnless_dataset(_):
    params = _maria_params()
    if params is None:
        pytest.skip("MariaDB connection not configured (.env / MARIA_DB_HOST)")
    from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler

    db_name = "p2_audit_maria_migration"
    dataset_path = Path(tempfile.mkdtemp(prefix="atrium_p2audit_maria_"))
    shutil.rmtree(dataset_path, ignore_errors=True)
    try:
        handler = MariaDBHandler(params["host"], params["user"], params["password"], db_name, params["port"])
        handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    except Exception as e:
        pytest.skip(f"MariaDB not reachable at {params['host']}:{params['port']} ({e})")

    cp = {"sqltype": "mariadb", "host": params["host"], "user": params["user"],
          "password": params["password"], "database": db_name, "port": params["port"]}
    sdk = AtriumSDK.create_dataset(dataset_location=dataset_path, database_type="mariadb",
                                   connection_params=cp)
    try:
        m = sdk.insert_measure(measure_tag="hr", freq=1.0, freq_units="Hz", units="bpm")
        # Simulate a pre-P2 Maria dataset: drop the two P2 columns.
        with sdk.sql_handler.connection() as (conn, cursor):
            cursor.execute("ALTER TABLE measure DROP COLUMN signal_kind")
            cursor.execute("ALTER TABLE measure DROP COLUMN value_type")
            conn.commit()
        sdk.close()

        # Reconnect with auto_upgrade -> the mirrored Maria ALTER must run.
        sdk2 = AtriumSDK(dataset_location=dataset_path, metadata_connection_type="mariadb",
                         connection_params=cp, auto_upgrade=True)
        try:
            info = sdk2.get_measure_info(m)
            assert (info["signal_kind"], info["value_type"]) == ("waveform", "numeric")
            # Idempotent re-run.
            assert sdk2.sql_handler.update_measure_schema() is False
            # First-write establishment works on the freshly migrated Maria schema.
            d = sdk2.insert_device(device_tag="dev")
            sdk2.write_time_value_pairs(m, d, times_for(3), NUM([1, 2, 3]))
            assert read_raw_value_type_maria(sdk2, m) == "numeric"
        finally:
            sdk2.close()
    finally:
        shutil.rmtree(dataset_path, ignore_errors=True)


def read_raw_value_type_maria(sdk_obj, measure_id):
    with sdk_obj.sql_handler.connection() as (conn, cursor):
        cursor.execute("SELECT value_type FROM measure WHERE id = ?", (int(measure_id),))
        return cursor.fetchone()[0]
