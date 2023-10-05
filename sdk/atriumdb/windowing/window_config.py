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

from typing import List, Tuple, Union


class WindowConfig:
    def __init__(self, measures: Union[List[str], List[Tuple[str, float, str]]], window_size_sec: Union[float, int],
                 window_slide_sec: float, allowed_lateness_sec: int, earliest_signal_time: int = None,
                 source_out_of_order: bool = True, offset: float = 0, filter_out_same_values: bool = True) -> None:
        """
        :param measures: List of measures, either a list of strings or list of tuples (str, float, str).
        :type measures: Union[List[str], List[Tuple[str, float, str]]]
        :param window_size_sec: Window size in seconds.
        :type window_size_sec: Union[float, int]
        :param window_slide_sec: Window slide time in seconds.
        :type window_slide_sec: float
        :param allowed_lateness_sec: Lateness allowed for the window to emit its data, in seconds (Live Streaming Only).
        :type allowed_lateness_sec: int
        :param earliest_signal_time: Earliest timestamp (in sec) of signals that will be allowed into a window stream.
                                     Default None value will allow processing of all arriving messages with a given routing key. (Live Streaming Only)
        :type earliest_signal_time: int
        :param source_out_of_order: A flag to indicate if the source is out of order. (Live Streaming Only)
        :type source_out_of_order: bool
        :param offset: Offset in seconds. Default is 0. (Live Streaming Only)
        :type offset: float
        :param filter_out_same_values: Controls whether to silently reject the windows completely consisting of a same value.
                                       Defaulting to True, as such windows are generally considered irrelevant. (Live Streaming Only)
        :type filter_out_same_values: bool
        """
        self.window_size_ns = int(window_size_sec * 10 ** 9)
        self.window_size_sec = float(window_size_sec)
        self.window_slide_ns = int(window_slide_sec * 10 ** 9)
        self.allowed_lateness_ns = int(allowed_lateness_sec * 10 ** 9)
        self.earliest_signal_time = earliest_signal_time
        if earliest_signal_time:
            self.earliest_signal_time = int(earliest_signal_time * 10 ** 9)
        self.source_out_of_order = source_out_of_order
        self.filter_out_same_values = filter_out_same_values
        self.offset = int(offset * 10 ** 9)

        # Validate all params. Measures are passed separately as they need to be validated before becoming an attribute
        self.validate(measures)

        self.measure_ids, self.measure_names = self.__get_measure_ids_and_names(measures)

    def validate(self, measures):
        # Check types
        if not isinstance(measures, list):
            raise ValueError("measures must be a list of strings or tuples")

        if not all(isinstance(x, (str, tuple)) for x in measures):
            raise ValueError("All elements in measure_names must be either strings or tuples")

        if not all(isinstance(x, str) or (isinstance(x, tuple) and len(x) == 3
                                          and isinstance(x[0], str)
                                          and isinstance(x[1], float)
                                          and isinstance(x[2], str))
                   or (isinstance(x, tuple) and len(x) == 3
                       and isinstance(x[0], str)
                       and isinstance(x[1], int)
                       and isinstance(x[2], str))
                   for x in measures):
            raise ValueError("All tuples in measure_names must contain 3 elements: (str, float, str)")

        if not isinstance(self.window_size_sec, float):
            raise ValueError("window_size_sec must be a float")

        if not isinstance(self.window_slide_ns, int):
            raise ValueError("window_slide_ns must be a int")

        if not isinstance(self.allowed_lateness_ns, int):
            raise ValueError("allowed_lateness_sec must be an int")

        if self.earliest_signal_time is not None and not isinstance(self.earliest_signal_time, int):
            raise ValueError("earliest_signal_time must be an int or None")

        if not isinstance(self.source_out_of_order, bool):
            raise ValueError("source_out_of_order must be a bool")

        if not isinstance(self.filter_out_same_values, bool):
            raise ValueError("filter_out_same_values must be a bool")

        # Check constraints
        if self.window_size_sec <= 0:
            raise ValueError("window_size_sec must be greater than 0")

        if self.window_slide_ns <= 0:
            raise ValueError("window_slide_ns must be greater than 0")

        if self.allowed_lateness_ns < 0:
            raise ValueError("allowed_lateness_ns must be greater than or equal to 0")

        if self.earliest_signal_time is not None and self.earliest_signal_time < 0:
            raise ValueError("earliest_signal_time must be greater than or equal to 0")

        if self.offset < 0:
            raise ValueError("offset cannot be negative")

    @staticmethod
    def __get_measure_ids_and_names(measures):
        """
        Convert given measures to Tuples of (Measurement Name: str, Frequency: float, Unit of Measure: str).
        If measure_name is a plain string, freq and uom are set to None, which will be interpreted by system as 'Any'
        Returns measure IDs as a valid list of tuples and measure names as a list of strings.
        """
        measure_names = list()
        measure_ids = list()
        for measure in measures:
            if isinstance(measure, str):
                measure_ids.append((measure, None, None))
                measure_names.append(measure)
            else:  # in case the measure is already provided as a valid tuple
                measure_ids.append(measure)
                measure_names.append(measure[0])  # Extract measure name as a first member of the tuple
        return measure_ids, measure_names
