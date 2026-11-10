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

import os
import warnings

import pytest

from atriumdb import block_wrapper
from atriumdb.block_wrapper import WrappedBlockDll


def test_forked_process_threads(monkeypatch):
    block = WrappedBlockDll("", num_threads=4)

    # No process has used several threads yet, as in a DataLoader worker forked before any read.
    monkeypatch.setattr(block_wrapper, "_THREAD_POOL_PID", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert block._get_num_threads() == 4
    assert block_wrapper._THREAD_POOL_PID == os.getpid()

    # The parent process used several threads before forking.
    monkeypatch.setattr(block_wrapper, "_THREAD_POOL_PID", os.getpid() + 1)
    with pytest.warns(UserWarning, match="Start worker processes"):
        assert block._get_num_threads() == 1
