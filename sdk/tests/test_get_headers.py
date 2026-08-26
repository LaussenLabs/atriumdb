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
get_headers / get_headers_from_blocks: the header-only read of a block range.

These two were written as module-level functions taking ``self`` and were never bound
to AtriumSDK, so ``sdk.get_headers(...)`` raised AttributeError for every caller. The
only test that called it sits in test_mit_bih's single-signal branch, and every mitdb
record is 2-channel, so it never ran and the breakage stayed invisible.

With the methods bound, two further bugs surfaced, both covered below:

  * ``get_headers_from_blocks`` read ``row[0], row[1]`` of a condensed read-list row as
    ``(file_id, start_byte)``. Those columns are ``(measure_id, device_id)`` -- the
    file_id is ``row[2]`` -- so the read asked for a nonexistent file id.
  * it condensed the block list first. Condensing merges byte-adjacent blocks into one
    larger read, which is right for reading whole blocks and wrong here: the caller
    slices the result at fixed header_size strides and counts on exactly one header per
    block, so a merged run silently lost headers.

The dataset below shrinks ``block_size`` and does ONE write, so the blocks land in a
single file byte-contiguously and ``condense_byte_read_list`` collapses all of them into
one read. That is the shape that makes the condensing bug visible; blocks spread over
several files never merge, and a test built that way passes with the bug still in place.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_get_headers.py -q
"""
import numpy as np

from atriumdb import AtriumSDK
from atriumdb.adb_functions import condense_byte_read_list

DB_NAME = 'atrium-get-headers'

FREQ_HZ = 1
PERIOD_NS = 1_000_000_000
BLOCK_SIZE = 32
NUM_VALUES = 256          # -> NUM_VALUES / BLOCK_SIZE blocks, all in one file


def _header_fields(header):
    """A BlockMetadata's fields as a plain dict. It is a bare ctypes Structure, so
    ``==`` on it compares identity, not contents."""
    return {name: getattr(header, name) for name, *_ in header._fields_}


def _build_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    # Small blocks + one write => many byte-contiguous blocks in a single file.
    sdk.block.block_size = BLOCK_SIZE

    measure_id = sdk.insert_measure('ecg', freq=FREQ_HZ, freq_units='Hz', units='mV')
    device_id = sdk.insert_device('dev_headers')

    times = np.arange(NUM_VALUES, dtype=np.int64) * PERIOD_NS
    values = np.arange(NUM_VALUES, dtype=np.float64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values, freq=FREQ_HZ, freq_units='Hz')

    end_time_n = NUM_VALUES * PERIOD_NS
    return sdk, measure_id, device_id, end_time_n


def test_get_headers_matches_get_data(dataset_for_backend):
    """get_headers returns exactly the headers a full get_data decode produces."""
    sdk, measure_id, device_id, end_time_n = _build_dataset(*dataset_for_backend(DB_NAME))

    headers, _times, values = sdk.get_data(measure_id, 0, end_time_n, device_id=device_id)
    assert values.size == NUM_VALUES
    # Both bugs need several blocks, and the condensing one needs them contiguous.
    assert len(headers) > 1, "expected several blocks; the regression needs them"
    assert len(condense_byte_read_list(
        sdk.sql_handler.select_blocks(int(measure_id), 0, int(end_time_n), device_id, None))) == 1, \
        "blocks must condense to ONE read, else the condensing bug is not exercised"

    just_headers = sdk.get_headers(measure_id, 0, end_time_n, device_id=device_id)

    assert len(just_headers) == len(headers)
    for just_header, header in zip(just_headers, headers):
        assert _header_fields(just_header) == _header_fields(header)


def test_get_headers_is_bound_to_the_sdk(dataset_for_backend):
    """A plain attribute lookup finds it -- this is what was broken."""
    sdk, _measure_id, _device_id, _end = _build_dataset(*dataset_for_backend(DB_NAME))

    assert callable(getattr(sdk, 'get_headers', None))
    assert callable(getattr(sdk, 'get_headers_from_blocks', None))


def test_get_headers_empty_range_returns_empty(dataset_for_backend):
    """A range with no blocks is an empty list, not an error."""
    sdk, measure_id, device_id, end_time_n = _build_dataset(*dataset_for_backend(DB_NAME))

    assert sdk.get_headers(measure_id, end_time_n * 10, end_time_n * 20, device_id=device_id) == []


def test_get_headers_from_blocks_directly(dataset_for_backend):
    """The lower-level entry point agrees with the full decode for the same blocks."""
    sdk, measure_id, device_id, end_time_n = _build_dataset(*dataset_for_backend(DB_NAME))

    block_list = sdk.sql_handler.select_blocks(int(measure_id), 0, int(end_time_n), device_id, None)
    assert len(block_list) > 1

    filename_dict = sdk.get_filename_dict(list({row[3] for row in block_list}))
    just_headers = sdk.get_headers_from_blocks(block_list, filename_dict)

    headers, _times, _values = sdk.get_data(measure_id, 0, end_time_n, device_id=device_id)
    assert len(just_headers) == len(headers)
    for just_header, header in zip(just_headers, headers):
        assert _header_fields(just_header) == _header_fields(header)
