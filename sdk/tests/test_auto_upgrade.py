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

"""W4-autoupgrade: the two gaps auto_upgrade did not close before this file --
the MariaDB insert_interval_union stored procedure never being created for an
existing dataset, and no dataset schema-version marker existing at all -- plus the
forward guard the marker makes possible (refusing to open a dataset stamped by a
newer AtriumDB than this one).

The measure-kind columns (period_ns/signal_kind/value_type) already migrate
correctly and are not this file's subject; test_measure_metadata_edge_cases.py
covers them. This file only re-touches them as part of building a convincing
"old-shaped" dataset for the end-to-end repair test.

Both backends via `_test_for_both` (see testing_framework.py). MariaDB needs
`atriumdb-mariadb` on :3306 (or MARIA_DB_HOST/PORT in .env); the SQLite half always
runs.
"""
import numpy as np
import pytest

from atriumdb import AtriumSDK
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler
from tests.testing_framework import _test_for_both

DB_NAME = 'auto_upgrade'
SEC = 1_000_000_000


def _make_old_shaped_dataset(db_type, dataset_location, connection_params):
    """Create a dataset with the current code, write one measure's worth of data,
    then strip everything auto_upgrade is responsible for restoring:
    the measure-kind columns, (MariaDB only) the insert_interval_union procedure,
    and the dataset schema-version marker. The closest re-creation of an older
    dataset available without keeping an old SDK build around.

    Returns (measure_id, device_id, times, values) for the caller to verify against
    after repair.
    """
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure(measure_tag='hr', freq=1.0, freq_units='Hz', units='bpm')
    device_id = sdk.insert_device(device_tag='dev')
    times = np.arange(10, dtype=np.int64) * SEC
    values = np.arange(10, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values, period=SEC, time_units='ns')
    sdk.close()

    sdk2 = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                      connection_params=connection_params)
    with sdk2.sql_handler.connection() as (conn, cursor):
        cursor.execute("ALTER TABLE measure DROP COLUMN signal_kind")
        cursor.execute("ALTER TABLE measure DROP COLUMN value_type")
        cursor.execute("DELETE FROM setting WHERE name = 'dataset_schema_version'")
        conn.commit()
    if db_type == 'mariadb':
        with sdk2.sql_handler.connection() as (conn, cursor):
            cursor.execute("DROP PROCEDURE IF EXISTS insert_interval_union")
            conn.commit()
    sdk2.close()

    return measure_id, device_id, times, values


def _probe_handler(db_type, dataset_location, connection_params):
    """A bare handler pointed at `dataset_location`, for read-only detection calls
    that must not go through AtriumSDK.__init__ -- which, on an old-shaped dataset
    with auto_upgrade=False, is exactly the raise under test elsewhere."""
    if db_type == 'mariadb':
        from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
        return MariaDBHandler(connection_params['host'], connection_params['user'],
                               connection_params['password'], connection_params['database'],
                               connection_params['port'])
    return SQLiteHandler(dataset_location / 'meta' / 'index.db')


# --------------------------------------------------------------------------- #
# End-to-end: detect, refuse, repair, re-detect, read old data back
# --------------------------------------------------------------------------- #
def test_detection_and_repair():
    _test_for_both(DB_NAME, _test_detection_and_repair)


def _test_detection_and_repair(db_type, dataset_location, connection_params):
    measure_id, device_id, times, values = _make_old_shaped_dataset(db_type, dataset_location, connection_params)

    # Detection is read-only and does not require constructing an AtriumSDK.
    probe = _probe_handler(db_type, dataset_location, connection_params)
    assert probe.get_dataset_schema_version() is None
    pending = probe.pending_schema_upgrades()
    if db_type == 'mariadb':
        assert probe.interval_union_procedure_current() is False
        assert 'the insert_interval_union stored procedure' in pending
    else:
        # SQLite's merge mode is plain Python, so an old SQLite dataset is missing
        # nothing functional and nothing is reported. An absent version stamp is not
        # itself a pending upgrade -- every dataset written before the stamp existed
        # lacks one, and listing it would make the whole installed base look broken.
        assert probe.interval_union_procedure_current() is True
        assert pending == []

    # This dataset is ALSO missing the measure columns, and those genuinely break
    # every measure query, so auto_upgrade=False refuses and names the fix. That
    # refusal comes from _reraise_missing_measure_column, not from the version
    # marker -- see test_dataset_missing_only_the_marker_still_opens for the
    # distinction, which is the whole reason the marker is not a gate.
    with pytest.raises(ValueError, match='auto_upgrade=True'):
        AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                  connection_params=connection_params, auto_upgrade=False)

    # auto_upgrade=True repairs it.
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                     connection_params=connection_params, auto_upgrade=True)
    assert sdk.sql_handler.pending_schema_upgrades() == []
    assert sdk.sql_handler.get_dataset_schema_version() == sdk.sql_handler.CURRENT_DATASET_SCHEMA_VERSION
    if db_type == 'mariadb':
        assert sdk.sql_handler.interval_union_procedure_current() is True

    # Data written before the upgrade still reads correctly afterwards.
    _, rt, rv = sdk.get_data(measure_id, 0, int(times[-1]) + SEC, device_id=device_id)
    assert rt.size == 10
    order = np.argsort(rt)
    np.testing.assert_array_equal(rt[order], times)
    np.testing.assert_array_equal(rv[order], values)
    sdk.close()

    # A second auto_upgrade=True run is a no-op, and the dataset now opens fine
    # even with the default auto_upgrade=False.
    sdk2 = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                      connection_params=connection_params, auto_upgrade=True)
    assert sdk2.sql_handler.pending_schema_upgrades() == []
    sdk2.close()

    sdk3 = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                      connection_params=connection_params)
    assert sdk3.sql_handler.pending_schema_upgrades() == []
    sdk3.close()


# --------------------------------------------------------------------------- #
# Unit-level: each repair primitive is independently idempotent
# --------------------------------------------------------------------------- #
def test_ensure_interval_union_procedure_idempotent():
    _test_for_both(DB_NAME + '_proc', _test_ensure_interval_union_procedure_idempotent)


def _test_ensure_interval_union_procedure_idempotent(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    if db_type != 'mariadb':
        # SQLite's merge mode is plain Python, not a stored procedure -- the hook
        # is a true no-op and costs nothing to call repeatedly.
        assert sdk.sql_handler.interval_union_procedure_current() is True
        assert sdk.sql_handler.ensure_interval_union_procedure() is False
        sdk.close()
        return

    # create_schema already created it -- repairing an already-current dataset
    # changes nothing.
    assert sdk.sql_handler.interval_union_procedure_current() is True
    assert sdk.sql_handler.ensure_interval_union_procedure() is False

    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("DROP PROCEDURE IF EXISTS insert_interval_union")
        conn.commit()
    assert sdk.sql_handler.interval_union_procedure_current() is False

    assert sdk.sql_handler.ensure_interval_union_procedure() is True
    assert sdk.sql_handler.interval_union_procedure_current() is True
    assert sdk.sql_handler._interval_union_proc_available is True
    # Repairing again is a no-op.
    assert sdk.sql_handler.ensure_interval_union_procedure() is False
    sdk.close()


def test_record_dataset_schema_version_idempotent():
    _test_for_both(DB_NAME + '_version', _test_record_dataset_schema_version_idempotent)


def _test_record_dataset_schema_version_idempotent(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    current = sdk.sql_handler.CURRENT_DATASET_SCHEMA_VERSION

    # create_dataset already stamped it.
    assert sdk.sql_handler.get_dataset_schema_version() == current
    assert sdk.sql_handler.record_dataset_schema_version() is False

    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("DELETE FROM setting WHERE name = 'dataset_schema_version'")
        conn.commit()
    assert sdk.sql_handler.get_dataset_schema_version() is None

    assert sdk.sql_handler.record_dataset_schema_version() is True
    assert sdk.sql_handler.get_dataset_schema_version() == current
    assert sdk.sql_handler.record_dataset_schema_version() is False
    sdk.close()


# --------------------------------------------------------------------------- #
# Forward guard: a dataset stamped by a NEWER AtriumDB refuses to open, under
# either auto_upgrade setting -- there is nothing this build could "upgrade" to
# close that gap, only pretend the dataset is older than it is.
# --------------------------------------------------------------------------- #
def test_newer_dataset_version_refuses_to_open():
    _test_for_both(DB_NAME + '_downgrade', _test_newer_dataset_version_refuses_to_open)


def _test_newer_dataset_version_refuses_to_open(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    newer = str(sdk.sql_handler.CURRENT_DATASET_SCHEMA_VERSION + 1)
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute("UPDATE setting SET value = ? WHERE name = 'dataset_schema_version'", (newer,))
        conn.commit()
    sdk.close()

    for auto_upgrade in (False, True):
        with pytest.raises(ValueError, match='newer AtriumDB'):
            AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                      connection_params=connection_params, auto_upgrade=auto_upgrade)


# --------------------------------------------------------------------------- #
# A freshly created dataset has nothing pending -- create_dataset's default
# auto_upgrade=False must not trip over its own brand-new schema.
# --------------------------------------------------------------------------- #
def test_fresh_dataset_has_no_pending_upgrades():
    _test_for_both(DB_NAME + '_fresh', _test_fresh_dataset_has_no_pending_upgrades)


def _test_fresh_dataset_has_no_pending_upgrades(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    assert sdk.sql_handler.pending_schema_upgrades() == []
    sdk.close()

    # Reopening with the default auto_upgrade=False must not raise.
    sdk2 = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                      connection_params=connection_params)
    sdk2.close()


# --------------------------------------------------------------------------- #
# The installed base: a dataset that is current in every way that matters and
# simply predates the version marker. This is every dataset ever written before
# the marker existed, and it must keep opening under the default
# auto_upgrade=False.
#
# An earlier version of this work listed the absent marker in
# pending_schema_upgrades() and raised on it, which took the entire existing
# installed base offline -- on SQLite, where there is no stored procedure to
# miss, an unstamped dataset was rejected for lacking a marker invented by the
# same change that rejected it.
# --------------------------------------------------------------------------- #
def test_dataset_missing_only_the_marker_still_opens():
    _test_for_both(DB_NAME + '_marker', _test_dataset_missing_only_the_marker_still_opens)


def _test_dataset_missing_only_the_marker_still_opens(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    measure_id = sdk.insert_measure('hr', 1, freq_units='Hz', units='bpm')
    device_id = sdk.insert_device('dev_marker')
    times = np.arange(10, dtype=np.int64) * SEC
    values = np.arange(10, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values)
    sdk.close()

    # Un-stamp it, leaving everything else exactly as a current SDK wrote it.
    probe = _probe_handler(db_type, dataset_location, connection_params)
    with probe.connection(begin=True) as (conn, cursor):
        cursor.execute("DELETE FROM setting WHERE name = ?",
                       (probe.DATASET_SCHEMA_VERSION_SETTING_NAME,))
    assert probe.get_dataset_schema_version() is None

    reopened = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                         connection_params=connection_params)
    _, read_times, read_values = reopened.get_data(
        measure_id, 0, int(times[-1]) + SEC, device_id=device_id)
    order = np.argsort(read_times)
    np.testing.assert_array_equal(read_times[order], times)
    np.testing.assert_array_equal(read_values[order], values)
    reopened.close()
