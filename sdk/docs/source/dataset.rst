Working with Datasets
========================

.. toctree::
   :maxdepth: 2

Iterator Usage
---------------------

Often we are interested in working with relatively small windows of data at a time. For visualizing, pre-processing
small amounts of data at once, or when we are training a model.

However the primary way of querying data, `AtriumSDK.get_data  <contents.html#atriumdb.AtriumSDK.get_data>`_ incurs an
overhead cost everytime it is called. This makes it an inefficient means of collecting a large amount of small windows
of data.

For this reason, AtriumDB has a `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ method, that
preloads large amounts of data in your RAM, and feeds it to you piece by piece in an iterable Class.

`AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ also does the job of windowing and indexing
your data for you, which makes tasks like training a model much simpler.

Dataset Iterator Example
############################

1. Initialize your SDK instance, connected to an existing dataset:

.. code-block:: python

    from atriumdb import AtriumSDK
    sdk = AtriumSDK(dataset_location=local_dataset_location)

2. Define the measures and patient or device cohorts:

**Example 1: By Patient**

.. code-block:: python

    from atriumdb import DatasetDefinition
    measures = ["MLII"]

    patient_ids = {
       1: "all",
       2: [{"time0": 1682739250000000000, "pre": 500000000, "post": 500000000}],
       3: [{"start": 1690776318966000000, "end": 1690777625288000000}],
       4: [{"start": 1690781225288000000}],
       5: [{"end": 1690787437932000000}],
    }

    definition = DatasetDefinition(measures=measures, patient_ids=patient_ids)

**Example 2: By Device**

.. code-block:: python

   device_ids = {
       1: "all",
       2: "all",
   }

   definition = DatasetDefinition(measures=measures, device_ids=device_ids)

You can also use mrns or device tags to device your sources. See the
`DatasetDefinition Class <contents.html#atriumdb.DatasetDefinition>`_ for more options.

3. Set your desired parameters: **window_duration** and **window_slide** (durations in nanoseconds by default,
changeable using ``time_units`` param, output times with conform to ``time_units`` units):

.. code-block:: python

   slide_size_nano = window_size_nano = 60_000_000_000  # 1 minute nano

4. Optional parameters:

**num_windows_prefetch** is the number of windows to preload for optimization, a higher number
increases efficiency at the cost of RAM usage (default will pick the number of windows such that the total number of
cached values is closest to 10 million values.)

**shuffle** When True, Randomizes the order of the dataset slices and the windows within a slice. You can define a slice size
using **cached_windows_per_source** which is the number of windows you want each slice to contain. Setting ``cached_windows_per_source=1``
ensures true randomness, but at great cost to the speed in which the windows are iterated.

**time_units** defines the time units of ``window_duration``, ``window_slide`` and ``gap_tolerance`` options are
``["s", "ms", "us", "ns"]``, default ``"ns"``.

Check the `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ documentation for a complete list of parameters

.. code-block:: python

   num_windows_prefetch = 100_000  # preload 100,000 windows before emitting
   gap_tolerance = 60  # Fill gaps in data less than 1 minute with nans
   time_units = "s"

5. Obtain the iterator:

.. code-block:: python

   iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano,
        num_windows_prefetch=num_windows_prefetch, gap_tolerance=gap_tolerance, time_units=time_units)

4. Iterate through the dataset:

.. code-block:: python

    for window_i, window in enumerate(iterator):
        print()
        print(window.start_time)
        print(window.device_id)
        print(window.patient_id)
        for (measure_tag, measure_freq_hz, measure_units), signal_dict in window.signals.items():
            print(measure_tag, measure_freq_hz, measure_units, signal_dict['measure_id'])
            print('times', signal_dict['times'])
            print('values', signal_dict['values'])
            print('expected_count', signal_dict['expected_count'])
            print('actual_count', signal_dict['actual_count'])

You can find explanations of the returned Window object in the :ref:`window_format` section below.

.. _window_format:

Window Format
#####################

The ``Window`` class represents a data structure for windowed data output by the
`DatasetIterator Class <contents.html#atriumdb.DatasetIterator>`_, it includes the raw
data organized into signal dictionaries, along with associated metadata, and additional
information related to patient and analysis results.

**Attributes**:

- ``signals`` : ``dict``
    A dictionary where each entry corresponds to a different measure signal, making it easier to handle measures of different frequencies. The keys of this dictionary are tuples, each consisting of the measure tag, the frequency of the measure (in Hz), and the units of the measure. The values are dictionaries containing metadata and data for each measure.

    Each signal dictionary has the following structure:

    - ``times`` : ``np.ndarray``
        A 1D numpy array representing the timestamps corresponding to each data point of the signal. This ensures that each data point in the window is associated with its precise capture time.

    - ``values`` : ``np.ndarray``
        A 1D numpy array containing the actual data points of the signal.

    - ``expected_count`` : ``int``
        An integer representing the expected number of data points in the signal window, based on its frequency and the window size.

    - ``actual_count`` : ``int``
        An integer representing the actual number of non-NaN data points in the signal window.

    - ``measure_id`` : ``int``
        An integer representing the unique identifier of the measure.

- ``start_time`` : ``int``
    The starting time, as a nanosecond epoch, of the window.

- ``device_id`` : ``int``
    An identifier representing the device from which the data was captured.

- ``patient_id`` : ``int``
    An identifier representing the patient associated with the data.

- ``label_time_series`` : ``np.ndarray``
    A 1D numpy array representing the labels for each data point in the window, typically used in supervised learning scenarios.

- ``label`` : ``np.ndarray``
    A 1D numpy array representing the aggregated or final label for the window, used for classification or regression outputs.

- ``patient_info`` : ``dict``
    A dictionary containing static patient meta information (such as id, mrn, gender, dob, etc.) returned by
    `AtriumSDK.get_patient_info`, as well as any dynamic fields requested in the `patient_history_fields` of `AtriumSDK.get_iterator`.
    This may include historical measurements like height and weight, along with their units and the timestamps they were recorded.


Example of the ``signals`` dictionary:

.. code-block:: python

    {
        ('heart_rate', 1, 'bpm'): {
            'times': np.array([1, 2, 3, ...]),
            'values': np.array([70, 71, 69, ...]),
            'expected_count': 100,
            'actual_count': 100,
            'measure_id': 123,
        },
        ('temperature', 0.01, 'C'): {
            'times': np.array([0, 10, 20, ...]),
            'values': np.array([36.6, 36.7, np.nan, ...]),
            'expected_count': 10,
            'actual_count': 9,
            'measure_id': 456,
        }
    }

Example of the ``patient_info`` dictionary:

.. code-block:: python

    {
        'id': 1,
        'mrn': 123456,
        'gender': 'M',
        'dob': 946684800000000000,  # Nanoseconds since epoch for date of birth
        'first_name': 'John',
        'middle_name': 'A',
        'last_name': 'Doe',
        'first_seen': 1609459200000000000,  # Nanoseconds since epoch
        'last_updated': 1609545600000000000,  # Nanoseconds since epoch
        'source_id': 1,
        'height': {  # Dynamic field example
            'value': 50.0,
            'units': 'cm',
            'time': 1609544500000000000,  # Nanoseconds since epoch
        },
        'weight': {  # Dynamic field example
            'value': 10.1,
            'units': 'kg',
            'time': 1609545500000000000,  # Nanoseconds since epoch
        }
    }


.. _aperiodic_windowing:

Aperiodic and String Measure Windowing
--------------------------------------

Windowing was originally designed for regularly sampled ``waveform`` measures, where every window is a
fixed-length grid of samples. Aperiodic measures (``sample``, ``event`` and ``state`` — see
:ref:`Measure Metadata <measure_metadata>`) and string measures do not naturally fit a fixed grid: they
have irregular timestamps and, for strings, non-numeric values. The iterator rasterizes these measures
onto the window grid using a per-measure **fill rule**, so every measure still produces a fixed-length
``values`` array you can stack into a tensor.

Fill rules by signal_kind
##########################

Each measure's ``signal_kind`` determines its default fill rule:

- ``waveform`` — unchanged. The existing NaN-filled sample grid is used; the fill configuration below does
  not apply.
- ``sample`` — **``carry_forward``** (default): each grid cell takes the most recent observed reading.
  Alternatives: ``sparse`` (only cells at an actual reading are filled, the rest are unknown) and
  ``aggregate:last|mean|min|max`` (reduce all readings falling in a cell). For string ``sample`` measures
  only ``carry_forward``, ``sparse`` and ``aggregate:last`` are valid (the numeric reductions
  ``mean|min|max`` are rejected).
- ``state`` — **``carry_forward``** with **left-censoring**: cells before the first observed state
  transition in the window are marked unknown (the true prior state is not known from within the window).
  A ``state`` value *is* carried across window boundaries within a region, so a long-running state stays
  populated in every window after the transition that set it.
- ``event`` — **``presence``** (default): each cell is ``1.0`` if any event occurred in it, else ``0.0``.
  Alternative: ``count`` (the number of events in the cell). Event cells are always "known" — absence is a
  meaningful ``0``, so there is no unknown sentinel for events.

.. important::

   **An** ``event`` **channel is occupancy, not identity.** ``presence`` and ``count`` are the *only*
   fill rules an ``event`` measure accepts — ``sparse``, ``carry_forward`` and ``aggregate:*`` all raise::

       ValueError: Fill rule 'sparse' is not valid for a 'event' measure.
       Allowed: ('presence', 'count').

   So an event measure holding ``ASYSTOLE``, ``V-TACH`` and ``SENSOR_OFF`` collapses in every window to a
   single "something happened" channel; you cannot recover *which* value fired. If you need per-cell
   identity, declare the measure ``signal_kind="state"`` (or ``"sample"``) instead — see
   :ref:`Choosing a signal_kind <choosing_signal_kind>` — or use one measure per event value.

.. warning::

   **Because event cells are always "known",** ``actual_count`` **always equals**
   ``expected_count`` **for an event channel.** The common quality-filter recipe below (and in
   :ref:`the tutorial <filtering_dataset_definitions>`) rejects windows on
   ``actual_count / expected_count``; that test can never fire for an event channel, so adding an event
   measure to a definition silently removes those windows from the filter's protection. Filter on the
   waveform / numeric channels you actually care about, keyed by measure id or ``signal_kind``.

Nominal raster period (1 s default)
####################################

Aperiodic measures are rasterized at a **nominal period**. Because the frequency stored on an aperiodic
measure does not describe a meaningful sampling grid, the period is resolved in this order:

1. a per-measure ``period_overrides[measure_id]`` (in ``time_units``), if given;
2. the measure's stored period, for ``waveform`` measures only;
3. otherwise a **1 second default** for aperiodic kinds.

The unknown-value sentinel (current limitation)
###############################################

"Unknown" cells — data gaps, ``sparse`` cells with no reading, and the left-censored region of a
``state`` measure before its first observed transition — are marked **with a sentinel in the ``values``
array**, not with a separate mask:

- ``NaN`` for numeric (float) channels;
- ``-1`` (``UNKNOWN_STRING_CODE``, which decodes to ``"<unknown>"``) for string / int64 code channels.

.. warning::

   A sentinel **conflates "unknown / censored" with a genuine missing reading** — a ``NaN`` in a window
   could mean "no data here" or "the recorded value was itself NaN". There is no separate ``known`` mask
   yet; a dedicated per-signal mask is a planned future enhancement. If your downstream logic must
   distinguish the two, keep this in mind.

Configuring fill on ``get_iterator``
####################################

Three `AtriumSDK.get_iterator <contents.html#atriumdb.AtriumSDK.get_iterator>`_ parameters control this
behaviour:

- ``aperiodic_fill`` — a **global default** fill rule applied to every aperiodic measure whose ``signal_kind``
  accepts it. A global default that is incompatible with a given measure's kind silently falls back to that
  kind's per-kind default (it never raises).
- ``fill_overrides`` — a ``{measure_id: rule}`` mapping of **per-measure** fill rules. Unlike
  ``aperiodic_fill``, an override that is incompatible with the measure's kind **raises** a ``ValueError``.
- ``period_overrides`` — a ``{measure_id: period}`` mapping of per-measure nominal raster periods, expressed
  in the iterator's ``time_units``.

.. note::

   ``fill_overrides`` and ``period_overrides`` are keyed by **measure id (int)**, while a
   ``DatasetDefinition`` is normally written in measure *tags*. Resolve the ids first with
   `AtriumSDK.get_measure_id <contents.html#atriumdb.AtriumSDK.get_measure_id>`_::

       nibp_id = sdk.get_measure_id(measure_tag="NIBP_SYS", freq=1.0, freq_units="Hz", units="mmHg")
       iterator = sdk.get_iterator(definition, 60, 60, time_units="s",
                                   fill_overrides={nibp_id: "sparse"})

   Passing a tag raises with a list of the definition's measures and their ids.

``get_iterator`` also accepts ``start_time`` and ``end_time`` (in the iterator's ``time_units``), which
bound the whole iteration in the same way as ``validate(start_time=..., end_time=...)``. Setting an
explicit ``end_time`` is the reliable way to stop windows being emitted past the end of your recording.

.. _reading_string_windows:

Reading string windows
#######################

Whether a string measure arrives in a window as decodable **codes** or as a numeric **occupancy** channel
depends entirely on its ``signal_kind``:

- ``sample`` and ``state`` string measures carry ``int64`` dictionary **codes** in
  ``window.signals[key]['values']``, not decoded strings — this keeps windows compact and tensor-friendly.
  Decode on demand with either accessor:

  - ``window.decode_string_signal(sdk, measure_key)`` — a method on the :ref:`Window <window_format>` object;
  - ``iterator.decode_window_strings(window, measure_key)`` — the equivalent accessor on the iterator.

  In both, ``measure_key`` is the ``(tag, freq_hz, units)`` tuple that keys ``window.signals``. The unknown
  sentinel (``-1``) decodes to ``"<unknown>"`` (override via the ``unknown_value`` argument).

- ``event`` string measures carry ``float64`` **occupancy** (``presence`` 0.0/1.0, or ``count``), not
  codes. They are **not decodable** — the accessors deliberately raise rather than fabricate vocabulary
  strings::

      ValueError: Cannot decode string codes for signal ('alarm_text', 1.0, 'alarm'): the window's
      values have dtype float64, not int64 dictionary codes. This channel is a numeric
      rasterization -- an 'event' measure filled with 'presence'/'count' holds occupancy floats ...

  There is no fill configuration that makes an ``event`` measure decodable; ``sparse`` and
  ``carry_forward`` are rejected for that kind. To read the text of an ``event`` measure, use
  `AtriumSDK.get_string_data <contents.html#atriumdb.AtriumSDK.get_string_data>`_ outside the iterator, or
  declare the measure ``state``/``sample`` in the first place (see
  :ref:`Choosing a signal_kind <choosing_signal_kind>`).

.. code-block:: python

    # A 'state' string measure -> int64 codes, decodable.
    codes = window.signals[("vent_mode", 1.0, "mode")]['values']       # dtype int64
    text = window.decode_string_signal(sdk, ("vent_mode", 1.0, "mode"))
    # -> array(['SIMV', 'SIMV', 'PRVC', ...], dtype=object)

    # An 'event' string measure -> float occupancy, NOT decodable.
    occupancy = window.signals[("alarm_text", 1.0, "alarm")]['values']  # dtype float64, 0.0/1.0

Example
#######

This end-to-end example creates a string ``sample`` measure and a numeric ``sample`` measure, writes a few
aperiodic points, then iterates and decodes a string window (see :ref:`Measure Metadata
<measure_metadata>` for more on ``signal_kind`` / ``value_type``).

.. code-block:: python

    import numpy as np
    from atriumdb import AtriumSDK, DatasetDefinition

    sdk = AtriumSDK(dataset_location=local_dataset_location)
    device_id = sdk.insert_device(device_tag="monitor_1")

    # A string 'sample' measure (device status text) and a numeric 'sample' measure (SpO2).
    status_id = sdk.insert_measure(measure_tag="device_status", freq=1.0, freq_units="Hz",
                                   units="status", signal_kind="sample", value_type="string")
    spo2_id = sdk.insert_measure(measure_tag="spo2", freq=1.0, freq_units="Hz",
                                 units="%", signal_kind="sample", value_type="numeric")

    # Write a handful of aperiodic points (seconds).
    sdk.write_time_value_pairs(status_id, device_id, np.array([2.0, 41.0]),
                               ["OK", "SENSOR_OFF"], time_units="s")
    sdk.write_time_value_pairs(spo2_id, device_id, np.array([0.0, 30.0, 55.0]),
                               np.array([98.0, 95.0, 91.0]), time_units="s")

    definition = DatasetDefinition(measures=["device_status", "spo2"], device_ids={device_id: "all"})

    iterator = sdk.get_iterator(
        definition,
        window_duration=60,           # 1 minute
        window_slide=60,
        time_units="s",
        aperiodic_fill="carry_forward",              # global default: carry the last value forward
        fill_overrides={spo2_id: "sparse"},          # spo2: only fill cells with an actual reading
        period_overrides={spo2_id: 5},               # rasterize spo2 every 5 s instead of the 1 s default
    )

    spo2_key = ("spo2", 1.0, "%")
    status_key = ("device_status", 1.0, "status")
    for window in iterator:
        # Numeric sample measure ('sparse'): NaN in every cell without an actual reading.
        spo2 = window.signals[spo2_key]['values']

        # String sample measure (carried forward): values are int64 codes -> decode on demand.
        status_codes = window.signals[status_key]['values']
        status_strings = window.decode_string_signal(sdk, status_key)   # "<unknown>" for sentinel cells
        # equivalently: iterator.decode_window_strings(window, status_key)
        print(spo2, status_codes, status_strings)

Current limitations
###################

- **Sentinel, not a mask** — as noted above, unknown/censored cells are only distinguishable by the
  sentinel value in ``values``; there is no separate ``known`` mask yet.
- **State rasterization is left-censored only** — within a window, a ``state`` channel marks cells before
  the first observed transition as unknown, but it does not itself bound the state on its *right* edge:
  the last observed value is carried to the end of the region. If you need an explicitly right-bounded
  state, derive it from
  `AtriumSDK.get_event_intervals <contents.html#atriumdb.AtriumSDK.get_event_intervals>`_, whose
  ``end_censored`` flag tells you when the closing marker was never seen — see
  :ref:`Building an "in state" channel <in_state_channel>`.
- **``carry_forward`` looks back a bounded distance, not an unbounded one** — at a region's start
  the iterator seeds ``carry_forward`` from the most recent reading in the **24 hours** preceding
  the definition's range start, so a region beginning mid-cycle for a slow-moving measure (NIBP
  every 5 minutes, a lab result) renders from the value actually in effect rather than from
  ``NaN``. A prior reading older than that horizon is not carried in, and those cells stay
  unknown. The horizon is measured from the definition's range start, not from a batch boundary,
  so rendered values do not change with ``num_windows_prefetch`` or batching.
- **The ``lightmapped`` iterator and the ``definition.filter`` path ignore the fill configuration** — they
  use the numeric grid path only and do not apply ``aperiodic_fill`` / ``fill_overrides`` /
  ``period_overrides``. Passing fill configuration to a ``lightmapped`` iterator emits a warning.
- **``Window`` carries no anchor identity** — windows arrive as a flat stream keyed only by
  ``start_time``; see :ref:`Recovering the anchor <event_anchored_regions>` for the recipe to map a window
  back to the event that produced it.

.. _in_state_channel:

Recipe: a 0/1 "in state A → B" channel
#######################################

Event pairing **is** implemented — see
`AtriumSDK.get_event_intervals <contents.html#atriumdb.AtriumSDK.get_event_intervals>`_ and
:ref:`Event Queries <event_queries>`. There is no API that turns a ``from → to`` pair directly into a
per-cell channel, but the iterator already gives you the grid times, so building one is a few lines. This
is the usual way to get an "is the patient in anesthesia right now" mask aligned with the waveform grid:

.. code-block:: python

    import numpy as np

    ivs = sdk.get_event_intervals(
        measure=event_measure_id,               # measure ID (int) - tags are not accepted here
        from_value="Anesthesia START", to_value="Anesthesia STOP",
        device_id=device_id, start_time=0, end_time=7200, time_units="s")

    key = ("anesthesia_events", 1.0, "event")
    for window in iterator:
        t = np.asarray(window.signals[key]['times'])     # nanosecond grid
        in_state = np.zeros(t.size, dtype=np.float32)
        for iv in ivs:                                   # times are ALWAYS nanoseconds
            in_state[(t >= iv['start_time_n']) & (t < iv['end_time_n'])] = 1.0
        print(window.start_time, in_state.mean())

Alternatively, if the same markers are declared as a ``state`` measure, the iterator produces the channel
directly as decodable ``int64`` codes with no extra work — see
:ref:`Choosing a signal_kind <choosing_signal_kind>`.

Iterator Types
------------------------

The `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ method supports three different types of iterators: default, filtered, and mapped. Each type serves different purposes and offers unique functionalities to handle your dataset windows as per your needs.

Default Iterator
####################

By default, or if you set `iterator_type` to None or `"iterator"`, you get the standard iterator. This returns an object that implements the `__iter__` and `__next__` methods, which allows you to iterate over a dataset's windows. The windows returned are :ref:`Window <window_format>` objects, which you can query for relevant signals, start time, device information, patient information, and any labels specified in your :ref:`dataset definition <definition_file_format>`.

Filtered Iterator
####################

The filtered iterator is similar to the default iterator, but it adds an additional filter functionality by accepting a user made filter function that decides whether a window should be included or skipped during iteration.

To use the filtered iterator, set `iterator_type` to `"filtered"` and pass a filter function using the `window_filter_fn` parameter. This filter function should take a window object as input and return `True` if the window should be included and `False` otherwise.

Example of defining a filter function:

.. code-block:: python

    def my_filter(window):
        # Your condition here
        return window.signals[("ECG_II", 500.0, "mV")]['actual_count'] >= 5  # at least 5 non-nan values.

    iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano, iterator_type='filtered', window_filter_fn=my_filter)

.. note::

    The filter function is a good place to do preprocessing. Any modifications made to the window object within the filter function will be retained when the window is passed back through the iterator.


Mapped Iterator
####################

The mapped iterator allows random access to dataset windows by using the `__getitem__` method. This means you can index the iterator directly to get a specific window, which is a useful feature if you need precise control over which windows to access, such as when labeling or visualizing specific windows by their indices.

To use the mapped iterator, set `iterator_type` to `"mapped"`. Be aware that this iterator type might be slower compared to the default iterator, as it cannot take advantage of certain optimizations related to sequential access.

Example of using the mapped iterator:

.. code-block:: python

    iterator = sdk.get_iterator(definition, window_size_nano, slide_size_nano, iterator_type='mapped')

    # Access a specific window by index
    window = iterator[5]
    print(window.start_time)

Recommendations
####################

- For most use cases, including model training and general window iteration, the default iterator should suffice. If you need to ensure data randomness for model training, you can set the `shuffle` parameter to `True`.
- Use the filtered iterator when you need to filter or preprocess windows on-the-fly.
- Use the mapped iterator for tasks that require random access to specific windows by their indices. However, note that it may be slower due to the lack of sequential access optimizations.

For further information and options on the `get_iterator` method, `check its section in the API Reference  <contents.html#atriumdb.AtriumSDK.get_iterator>`_.

.. _definition_file_format:

Definition File Format
------------------------------

Source Types
#################

The YAML file defines various source types (``patient_ids``, ``mrns``, ``device_ids``, and ``device_tags``). For each source type, specific sources or entities are identified by unique names.

Time Entries
#################

For each source/entity, you can provide multiple time entries. Each time entry describes a relevant time period or event for that source. There are four types of time specifications:

1. **Interval-based**: This type specifies a continuous interval with a ``start`` and/or ``end`` time.

   - ``start``: The beginning of the interval (nanosecond Unix Epoch Time).
   - ``end``: The end of the interval (nanosecond Unix Epoch Time).

2. **Event-based**: This type specifies an event time and the time period before and after the event.

   - ``time0``: The exact time of the event (nanosecond Unix Epoch Time).
   - ``pre``: Duration before the event (in nanoseconds).
   - ``post``: Duration after the event (in nanoseconds).

3. **All** All available time data can be specified using the ``all`` keyword.

4. **Event-anchored**: Instead of a fixed timestamp, build the region(s) *around string
   (event) values* — every occurrence of an event value, or the span between an opening and
   a closing event. See :ref:`event_anchored_regions` for the ``anchor`` and ``from``/``to``
   forms.

Measures
#################

The ``measures`` section lists the measures you want your Dataset to contain. Each measure can either be:

1. The measure tag, if there is only one measure with that tag.
2. A complete measure triplet which includes:

   - ``tag``: The tag identifying the measure.
   - ``freq_hz`` or ``freq_nhz``: The frequency of the measure in Hertz (floating) or nanoHertz (integer).
   - ``units``: The unit of the measure (e.g., volts, bpm).

Labels
#################

The ``labels`` section lists the names of the labels you want to include in your dataset.


.. code-block:: yaml

   # could be mrns, device_ids or device_tags
   patient_ids:
        12345:
            - start: 1682739200000000000  # nanosecond Unix Epoch Time
                end: 1682739300000000000    # nanosecond Unix Epoch Time
            - time0: 1682739250000000000   # nanosecond Unix Epoch Time
                pre: 500000000               # nanoseconds before the event_time
                post: 500000000              # nanoseconds after the event_time
        67890: all
        11111:
            - start: 1682739200000000000  # Start with no end

   measures:
        - heart_rate
        - tag: ECG
          freq_hz: 62.5
          units: mV
        - tag: ABP
          freq_nhz: 250000000000
          units: mV

    labels:
         - sinus_rhythm
         - atrial_fibrillation
         - noise_artifact


Dataset Definitions
-----------------------------

Creating a DatasetDefinition object
###################################

You can create a `DatasetDefinition <contents.html#atriumdb.DatasetDefinition>`_ object in several ways:

1. Reading from an existing YAML file:

   .. code-block:: python

      dataset_definition = DatasetDefinition(filename="/path/to/my_definition.yaml")

2. Creating a definition by specific devices:

   .. code-block:: python

      device_tags = {"tag_1": [{'start': start_time_nano_1, 'end': end_time_nano_1}], "tag_2": [{'time0': event_time_nano_2, 'pre': nano_before_event_2, 'post': nano_after_event_2}]}
      labels = ["atrial_fibrillation", "sinus_rhythm", "noise_artifact"]
      dataset_definition = DatasetDefinition(
            measures=measures,
            device_tags=device_tags,
            labels=labels
        )


3. Creating a definition by specific patients:

   .. code-block:: python

      patient_ids = {1234567: [{'start': start_time_nano_1, 'end': end_time_nano_1}], 7654321: "all"}
      dataset_definition = DatasetDefinition(measures=measures, patient_ids=patient_ids)


.. _event_anchored_regions:

Event-Anchored Regions
######################################

In addition to fixed ``start``/``end`` intervals and ``time0`` events, a source's time-spec
list may contain **event-anchored regions**. These build time ranges *around string (event)
values* — for example "5 minutes around every Anesthesia START", or "between each START and
the next STOP" — resolved at :py:meth:`~DatasetDefinition.validate` time against the actual
events recorded for that source. See :ref:`Event Queries <event_queries>` for the underlying
event model (string measures, the ``from → to`` pairing, and the ``within`` cascade).

There are two forms. Both require a ``measure`` naming the **event (string) measure** to look
in, given as a measure tag (string) or a measure id (int). A numeric measure raises an error.

**(a)** ``anchor`` **— a window around every occurrence.**
For each occurrence of the ``anchor`` value in the event measure, a window
``[t - pre, t + post]`` is emitted:

- ``anchor``: the event value to anchor on (must be in the measure's vocabulary).
- ``measure``: the event/string measure tag or id.
- ``pre``: nanoseconds before each occurrence (optional, default ``0``).
- ``post``: nanoseconds after each occurrence (optional, default ``0``).
- ``within``: optional container scoping (see below).

**(b)** ``from``/``to`` **— a region between an opening and a closing event.**
Each interval between a ``from`` event and the next ``to`` event (using the same *collapse*
pairing as `AtriumSDK.get_event_intervals <contents.html#atriumdb.AtriumSDK.get_event_intervals>`_)
becomes a region:

- ``from`` / ``to``: the opening and closing event values (both required; both must be in the
  measure's vocabulary).
- ``measure``: the event/string measure tag or id.
- ``within``: optional container scoping (see below).
- ``pre`` / ``post``: optional nanosecond padding applied to each derived interval
  (default ``0``).
- ``max_duration``: optional cap (nanoseconds) on each region's length.
- ``on_censored``: how to handle intervals whose true boundary lies outside the observed data
  (a ``from`` with no following ``to``, or a ``to`` with no preceding ``from``) — one of
  ``"clip"`` (default: clip the censored end to the container/range boundary, keep the region,
  and warn), ``"drop"`` (omit any censored interval), or ``"keep"`` (keep it unchanged).

In both forms the emitted windows are clipped to the validation ``start_time``/``end_time``
bounds. They are resolved from the **event measure's** own availability, not from the
availability of the measures you are asking for — an anchored region will happily cover a
stretch where the ECG you requested has no data, and those windows come back all-``NaN``.

.. note::

   ``on_censored="clip"`` (the default) clips a censored interval to the container boundary. When
   an **event** measure is what defines that boundary, the boundary used is its last *real* event
   timestamp, not the padded end of its availability index — an event is instantaneous, and the
   span between the final event and the index's padded end contains no event by construction. A
   ``from`` event that never closes therefore stops at the last observed event rather than running
   past the end of the recording.

   Two habits still make censored regions predictable, and both are worth keeping:

   - pass an explicit ``end_time`` to :py:meth:`~DatasetDefinition.validate` (or to
     ``get_iterator``) that reflects the real end of your data, or use ``on_censored="drop"`` to
     omit unterminated intervals altogether; and
   - declare an explicit ``freq``/``period`` when writing the event measure, so its presence index
     is not sized from a guessed period.

.. note::

   **Event-anchored regions are anchor-only** — they only define *time ranges*. The event
   measure itself is **not** added to the returned data automatically. To also receive the
   event channel in each window, add the event measure to the definition's ``measures``.

.. note::

   **Recovering the anchor.** Windows arrive as a flat stream; a ``Window`` exposes
   ``start_time``, ``device_id``, ``patient_id``, ``patient_info``, ``signals``, ``label`` and
   ``label_time_series`` — there is no ``anchor_time`` or region index. To label a window
   relative to its anchor ("t−2 min from START #3"), re-derive the anchors and bucket by
   ``start_time``:

   .. code-block:: python

      times, values = sdk.get_string_data(event_measure_id, start_time_n=start_n,
                                          end_time_n=end_n, device_id=device_id)
      anchors = sorted(int(t) for t, v in zip(times, values) if v == "Anesthesia START")

      for window in iterator:
          # the anchor whose [t-pre, t+post] region contains this window
          anchor = max((a for a in anchors if a - pre_ns <= window.start_time <= a + post_ns),
                       key=lambda a: -abs(a - window.start_time), default=None)
          offset_s = None if anchor is None else (window.start_time - anchor) / 1e9

**The** ``within`` **cascade.** When given, ``within`` scopes the emitted ranges to a
container. It follows the same cascade as
`AtriumSDK.get_event_intervals <contents.html#atriumdb.AtriumSDK.get_event_intervals>`_:
``"device_patient"`` → ``"encounter"`` → ``"none"`` (whole-stream). Omit ``within`` to leave
the ranges unscoped.

**Validation.** Unknown values raise at :py:meth:`~DatasetDefinition.validate` time: an event
``measure`` tag/id that does not exist, a ``measure`` that is not a string measure, or an
``anchor``/``from``/``to`` value not in that measure's vocabulary. A source that simply has no
matching events is not an error — it warns and contributes no ranges.

Structural problems, by contrast, raise from ``DatasetDefinition.__init__`` — for example a
region with an ``anchor``/``from``/``to`` but no ``measure`` key::

    ValueError: device_id 1: an event-anchored region requires a 'measure'
                (the event/string measure tag or id)

If you are wrapping definition construction in error handling, wrap the constructor as well as
``validate()``.

.. note::

   The ``measure`` key of an event-anchored region accepts a **tag or an id**.
   `AtriumSDK.get_event_intervals <contents.html#atriumdb.AtriumSDK.get_event_intervals>`_, the
   standalone query behind it, accepts a **measure id only** — passing a tag raises
   ``ValueError: invalid literal for int() with base 10: '...'``. Resolve it with
   ``sdk.get_measure_id(...)`` first.

**Example.** Two cohorts against the ``anesthesia_events`` string measure: one 5-minute window
around every ``"Anesthesia START"``, and one covering each ``"Anesthesia START"`` → next
``"Anesthesia STOP"`` span scoped to the encounter. The event measure is included in
``measures`` so the event channel is returned alongside ``ECG``.

.. code-block:: python

    from atriumdb import AtriumSDK, DatasetDefinition

    sdk = AtriumSDK(dataset_location=local_dataset_location)

    five_min_ns = 5 * 60 * 1_000_000_000  # 5 minutes in nanoseconds

    definition = DatasetDefinition(
        # "anesthesia_events" is included so the event channel is returned in each window;
        # without it, the regions below would still define the time ranges (anchor-only).
        measures=["ECG", "anesthesia_events"],
        device_ids={
            25: [
                # (a) 5 minutes around EVERY "Anesthesia START" occurrence.
                {
                    "anchor": "Anesthesia START",
                    "measure": "anesthesia_events",
                    "pre": five_min_ns,
                    "post": five_min_ns,
                },
                # (b) between each "Anesthesia START" and the next "Anesthesia STOP",
                #     scoped to the encounter, capping any single region at 6 hours.
                {
                    "from": "Anesthesia START",
                    "to": "Anesthesia STOP",
                    "measure": "anesthesia_events",
                    "within": "encounter",
                    "max_duration": 6 * 60 * 60 * 1_000_000_000,
                    "on_censored": "clip",
                },
            ],
        },
    )

    definition.validate(sdk=sdk)

The same YAML form:

.. code-block:: yaml

   measures:
        - ECG
        - anesthesia_events

   device_ids:
        25:
            - anchor: "Anesthesia START"
              measure: "anesthesia_events"
              pre: 300000000000    # 5 minutes in nanoseconds
              post: 300000000000
            - from: "Anesthesia START"
              to: "Anesthesia STOP"
              measure: "anesthesia_events"
              within: encounter
              max_duration: 21600000000000   # 6 hours in nanoseconds
              on_censored: clip

.. note::

   ``pre``, ``post`` and ``max_duration`` are always **nanoseconds**. Event-anchored regions
   can be mixed freely with the classic interval / ``time0`` / ``all`` specs in the same
   source's time-spec list.


Building Dataset Definitions from Intervals
###########################################

You can build a dataset definition directly from the measure or label availability using the :py:meth:`DatasetDefinition.build_from_intervals` class method.

This method analyzes available data (via the SDK) and constructs a time-region map for each source by finding regions where the specified signals or labels exist. This approach supports two key use cases:

1. **Measure-based construction**: Specify one or more measures and choose between:

   - **Union** (default): Includes all time intervals where *at least one* of the measures is available.
   - **Intersection**: Includes only those intervals where *all* requested measures are simultaneously available.

2. **Label-based construction**: Specify one or more labels to construct intervals around their presence in the data.

Additional parameters such as `start_time`, `end_time`, and `gap_tolerance` allow precise control over which intervals are included and how gaps in data are handled.

.. code-block:: python

   # Crease a dataset where ECG and ABP are simultaneously available.
   dataset_definition = DatasetDefinition.build_from_intervals(
       sdk=my_sdk,
       build_from_signal_type="measures",
       measures=["ECG", "ABP"],
       start_time=1735845426 * 10**9,
       end_time=1737236445 * 10**9,
       gap_tolerance=1_000_000_000,
       merge_strategy="intersection"
   )

Validating Dataset Definitions
###############################

Dataset definitions must be validated before use with most SDK-based operations.

Calling :py:meth:`DatasetDefinition.validate` performs the following:

- Confirms the existence of the requested measures and labels.
- Finds the intersection between the set of requested data and the set of available data.
- Stores the mapping of data sources and validated intervals in a structured dictionary.

SDK methods that consume a DatasetDefinition will automatically trigger this validation, and will not trigger it more than once, so you can still submit unvalidated definitions to the SDK.

.. code-block:: python

   dataset_definition.validate(
       sdk=my_sdk,
       gap_tolerance=5,
       measure_tag_match_rule="best",
       start_time=1735845426,
       end_time=1737236445,
       time_units="s"
   )

.. note::

   Repeated validation calls are avoided once the dataset is marked as validated internally via the ``is_validated`` flag.

Filtering Dataset Definitions
##############################

Once a DatasetDefinition is validated, you can filter it further using the :py:meth:`DatasetDefinition.filter` method.

Filtering works by sliding a window over the time ranges in the validated dataset. Each window is passed to your filter function, which returns `True` (keep) or `False` (discard). This can be used to exclude windows with missing data, poor signal quality, insufficient label coverage, etc.

Your filter function receives a :ref:`window_format` object and should return a boolean:

.. code-block:: python

   def my_filter_fn(window):
       # Accept only windows where at least 5 samples of the signal are available
       if window.signals[("ECG", 62.5, "mV")]['actual_count'] > 5:
            return False
       avg_value = window.signals[("ECG", 62.5, "mV")]['values'].mean()
       return avg_value > 0.3

   dataset_definition.filter(
       sdk=my_sdk,
       filter_fn=my_filter_fn,
       window_duration=1,
       window_slide=1,
       time_units='s'
   )

You may also control the inclusion of partial windows and adjust other advanced options like label thresholds or custom patient history fields.

Saving Dataset Definitions
###########################

You can save any DatasetDefinition to disk for later reuse using the :py:meth:`DatasetDefinition.save` method.

There are two supported formats:

- **YAML (.yaml or .yml)**: Saves the raw user-specified dictionary (`data_dict`). This format is suitable for editing or inspection.
- **Pickle (.pkl)**: Saves the fully validated and/or filtered definition, including all computed metadata. This format is faster for reuse and avoids re-validation.

.. code-block:: python

   dataset_definition.save("definition.yaml")  # Save original definition
   dataset_definition.validate(sdk=my_sdk)
   dataset_definition.save("validated_definition.pkl")  # Save validated version


Combining Dataset Definitions
##############################

The :py:func:`combine_definitions` function merges two or more :class:`DatasetDefinition` objects into a single
definition. Measures and labels are deduplicated, and source dictionaries (patients, devices, etc.) are merged
by taking the union of their time regions.

This is useful when you have separately constructed definitions — for example, one per patient cohort or one per
signal type — and need to unify them into a single definition for iteration or partitioning.

All input definitions must share the same validation status. If all are validated, the combined result is also
validated with the merged data. If the definitions have mixed validation status, a ``ValueError`` is raised;
either validate all of them first, or combine only unvalidated definitions and re-validate afterward.

.. code-block:: python

   from atriumdb import DatasetDefinition, combine_definitions

   # Two separately constructed definitions
   ecg_def = DatasetDefinition(
       measures=[{"tag": "ECG", "freq_hz": 500, "units": "mV"}],
       patient_ids={1: "all", 2: "all"},
       labels=["sinus"]
   )
   abp_def = DatasetDefinition(
       measures=[{"tag": "ABP", "freq_hz": 125, "units": "mmHg"}],
       patient_ids={2: "all", 3: "all"},
       labels=["hypotension"]
   )

   # Combine into one
   combined = combine_definitions([ecg_def, abp_def])
   # Or equivalently:
   combined = ecg_def.combine(abp_def)

   # Result has both measures, both label sets, and patients 1, 2, 3
   print(combined.data_dict['measures'])
   # [{'tag': 'ECG', 'freq_hz': 500, 'units': 'mV'}, {'tag': 'ABP', 'freq_hz': 125, 'units': 'mmHg'}]

.. note::

   When combining validated definitions, any previous filtering metadata is reset. Re-filter the combined
   definition if needed.


Partitioning Dataset Definitions
################################

The :py:func:`partition_dataset` function is used to split a validated `DatasetDefinition` into multiple partitions—commonly for
training, validation, and testing in machine learning workflows. It ensures that no patient appears in more than one partition,
which helps prevent data leakage across splits. The function also attempts to balance the distribution of selected labels
(defined via `priority_stratification_labels`) proportionally across the partitions according to the provided `partition_ratios`.

The partitioning process is stochastic by default. This means that repeated calls to the function can result in different splits,
even with the same inputs. To improve consistency and reproducibility, you can pass a fixed integer seed via the `random_state` argument.
If you want to explore several possible splits and choose the most balanced one, you can set `n_trials` to perform multiple partitioning attempts.
If `verbose=True`, a summary of each attempt is printed, including label distribution and the `random_state` value used.
You can then reuse the best-performing seed in future runs by passing it to `random_state`.

You can also specify `additional_labels`, which are not used to guide the partitioning but will be included in the partition summary reports.
This can be helpful for evaluating how well these labels are represented in each split.

Example:

.. code-block:: python

   from atriumdb import DatasetDefinition, AtriumSDK, partition_dataset

   definition = DatasetDefinition(
       measures=["ECG", "ABP"],
       device_ids={1: "all", 2: "all"},
       labels=["atrial_fibrillation", "sinus_rhythm"]
   )
   # Validate the definition first
   definition.validate(sdk=my_sdk)

   # Partition into train/val/test using stratified label balancing
   train_def, val_def, test_def = partition_dataset(
       definition,
       sdk=my_sdk,
       partition_ratios=[60, 20, 20],
       priority_stratification_labels=["label1", "label2"],
       random_state=42,
       verbose=True
   )

   # Or, explore 10 candidate partitions and display the 3 best
   (train_def, val_def, test_def), info = partition_dataset(
       definition,
       sdk=my_sdk,
       partition_ratios=[60, 20, 20],
       priority_stratification_labels=["label1", "label2"],
       n_trials=10,
       num_show_best_trials=3,
       verbose=True
   )


Cross-Validation
#################

The :py:func:`cross_validate_dataset` function generates cross-validation fold assignments from a
:class:`DatasetDefinition`. It partitions the dataset into ``n_folds`` equal-sized folds (by patient),
then enumerates every unique assignment of folds to training, validation, and test roles. For each
assignment, the constituent folds are recombined into single :class:`DatasetDefinition` objects.

This is a higher-level convenience built on top of :func:`partition_dataset` and
:func:`combine_definitions`. The fold partitioning happens once, and the same folds are reused across
all combinations — only the role assignments change.

.. code-block:: python

   from atriumdb import cross_validate_dataset

   folds = cross_validate_dataset(
       definition,
       sdk=my_sdk,
       n_folds=5,
       n_val_folds=1,
       n_test_folds=1,
       random_state=42,
       output_dir="./cv_output",
       filename_prefix="2025-06-01_",
       priority_stratification_labels=["atrial_fibrillation"],
       verbose=True,
   )

   # 20 unique combinations (5 choices for val × 4 remaining choices for test)
   for combo in folds:
       train_def = combo["train"]
       val_def = combo["val"]
       test_def = combo["test"]
       print(f"Train folds: {combo['fold_indices']['train']}, "
             f"Val folds: {combo['fold_indices']['val']}, "
             f"Test folds: {combo['fold_indices']['test']}")

Each combination is a dictionary with keys ``"train"``, ``"val"``, and ``"test"`` mapping to
:class:`DatasetDefinition` objects, plus ``"fold_indices"`` indicating which fold numbers were assigned
to each role.

When ``output_dir`` is provided, YAML definition files are saved for every combination:

.. code-block:: text

   cv_output/
     2025-06-01_dataset_def_train_fold_1.yaml
     2025-06-01_dataset_def_val_fold_1.yaml
     2025-06-01_dataset_def_test_fold_1.yaml
     2025-06-01_dataset_def_train_fold_2.yaml
     ...

The ``n_val_folds`` and ``n_test_folds`` parameters control how many folds are assigned to each role.
For example, with ``n_folds=5, n_val_folds=1, n_test_folds=1``, each combination uses 1 fold for
validation, 1 for testing, and the remaining 3 for training.




Exporting Datasets
----------------------------

Copying an Existing Dataset
###########################

You can duplicate an entire AtriumDB dataset, or a smaller subset, into a new location with minimal configuration. This is useful for:

- Exporting targeted data for experimentation or research
- Creating backups
- Distributing compute away from production systems
- Migrating data to new hardware or storage environments

By default, `transfer_data` reuses existing waveform blocks (`reencode_waveforms=False`), which is faster and conserves disk I/O.

.. code-block:: python

   from atriumdb import AtriumSDK, DatasetDefinition, transfer_data

   # 1. Open your source and create the export destination
   main_sdk = AtriumSDK(
       dataset_location=main_dataset_location,
       connection_params=main_connection_params,
       metadata_connection_type="mariadb"
   )
   export_sdk = AtriumSDK.create_dataset(
       dataset_location="/export/path"
   )

   # 2. Mirror all devices and select measures of interest
   all_devices = main_sdk.get_all_devices().keys()
   device_ids = {did: "all" for did in all_devices}
   measures = [("MDC_ECG_AMPL_ST_II", 200, "MDC_DIM_MICRO_VOLT")]

   # 3. Build the dataset definition
   full_def = DatasetDefinition(device_ids=device_ids, measures=measures)

   # 4. Copy the data using fast block reuse (no re-encoding)
   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=full_def,
       reencode_waveforms=False
   )

   # ——> All waveform ECGs and metadata are now copied to "/export/path"

.. note::

   Block reuse is preferred in most cases. Use `reencode_waveforms=True` only when:
   - You want to change block size (e.g., `dest_sdk.block.block_size`)
   - The source dataset has very small or fragmented blocks you'd like to consolidate

Exporting a Specific Patient Cohort
###################################

For targeted research or smaller exports, you can define a subset of MRNs or patient IDs:

.. code-block:: python

   # Export only selected patients
   cohort_mrns = [12345, 67890]
   mrns = {mrn: "all" for mrn in cohort_mrns}
   cohort_def = DatasetDefinition(
       mrns=mrns,
       measures=measures
   )

   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=cohort_def,
       reencode_waveforms=False
   )

   # ——> Only ECGs for patients 12345 and 67890 are copied

De-identification & Time Shifting
#################################

If you're sharing data externally (e.g., for research or compliance), you can enable:

- `deidentify`: Removes patient-level metadata (name, MRN, DOB, etc.) and scrambles patient IDs, True, False or a csv filepath which resolves to True and writes a csv containing the original to deidentified patient mapping.
- `time_shift`: Uniformly shifts all timestamps within the dataset to obscure precise dates within the patient information.

These can be used independently, but are often combined for data privacy.

.. important:: **Scope of de-identification**

   De-identification in AtriumDB covers **patient-level PHI and time-shifting**. Concretely,
   that is: the patient table's identifying columns (name, MRN, DOB, …), the patient ID
   remap, the ``encounter.visit_number`` scramble, and the uniform ``time_shift`` applied to
   every time-bearing column. ``log_hl7_adt`` is never transferred at all.

   It does **not** alter signal or label content. In particular, none of the following are
   changed by ``deidentify=True``:

   - **Numeric sample values** — unchanged, as you would expect.
   - **String (event) measure values** — unchanged. A string measure's values are signal
     data in exactly the way a numeric measure's samples are; ``deidentify=True`` will not
     replace "SIMV" or an alarm string with a placeholder.
   - **Label names, label text and label sources** — unchanged. Only a label's device and
     measure IDs are remapped and its times shifted.
   - **Bed, unit and institution names** — unchanged. A location name says *where* a
     recording happened, not *whose* it is.

   The principle is that if a caller is permitted to read a signal, they read all of it.
   De-identification is not a content filter over the data, and it does not silently
   redact it. If you have a specific free-text measure you want scrubbed anyway, ask for it
   explicitly with ``string_value_policy`` (see :ref:`string_value_policy_label`); it is
   never applied on your behalf.

.. code-block:: python

   # Scramble patient IDs and shift timestamps by 2 hours
   two_hours_s = 2 * 60 * 60
   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=cohort_def,
       deidentify="patient_mapping_file.csv",
       time_shift=two_hours_s,
       time_units="s"
   )

   # ——> Timestamps are shifted; a CSV maps old → new patient IDs

.. note::

   The mapping file lets you reconcile anonymized patient IDs if needed in the future. It is a
   **headerless** CSV of ``source_patient_id,destination_patient_id``, one pair per line::

       1,10004
       2,10000
       3,10003

   Store it separately from the shared extract — it is the key that undoes the pseudonymization.

.. important::

   ``deidentify`` handles patient-level metadata, ``encounter.visit_number`` and time-shifting.
   It does **not** by itself make a shared extract free of identifying text: signal content
   (including string measure values), label names and device tags all transfer **verbatim**, by
   design. See :ref:`String / event measure values <string_value_policy>` and
   :ref:`Other text surfaces <other_free_text_surfaces>` below before sharing.

.. _transferring_encounters:

Transferring Encounters and De-identification
#############################################

Beyond waveforms and patient information, ``transfer_data`` also carries the **encounter
family** — the ``encounter`` and ``device_encounter`` records that tie a patient/device to
an admission, along with the ``bed``, ``unit``, and ``institution`` rows they reference (the
location hierarchy is copied for referential integrity). This is what makes admission-scoped
features such as ``within: encounter`` work on the destination dataset.

The encounter family transfers **by default**. Set ``include_encounters=False`` to skip it.

.. note::

   ``log_hl7_adt`` (the raw HL7 ADT message log) is **never** transferred, regardless of
   ``include_encounters`` or the de-identification setting. There is currently no opt-in to
   include it.

.. _string_value_policy:
.. _string_value_policy_label:

String / event measure values
******************************

String-valued measures (events, states, free-text) transfer as first-class data in the native
``tsc`` format: the ``value_type`` / ``signal_kind`` measure metadata is carried across, event
timing and the interval index are preserved, and the per-measure dictionaries are reproduced in
the destination. When the destination already has a dictionary for that measure the vocabularies
are unioned and the transferred codes are remapped into the destination's code space.

**String measure values transfer verbatim in every mode,** ``deidentify=True`` **included.**
They are signal data in exactly the way a numeric measure's samples are: nobody expects
``deidentify=True`` to null out a heart rate, and nothing replaces a ventilator mode or an
alarm name either. Event queries and event-anchored definitions therefore keep working on the
destination without any extra configuration.

If you nonetheless have one particular free-text measure you want scrubbed on the way out,
``string_value_policy`` lets you ask for that **explicitly**. It is never applied on your behalf,
and ``deidentify`` has no influence on its default:

.. list-table:: ``string_value_policy``
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Effect
   * - ``"transfer"``
     - Copy the strings verbatim. **The default, always** — in every mode, ``deidentify=True``
       included.
   * - ``"redact"``
     - Write every string value as ``"<redacted>"``. Event timing, counts and the interval index
       survive; the text does not.
   * - ``"skip"``
     - Do not transfer any string measure data. The measures are still created in the
       destination, but with no rows and an empty vocabulary.
   * - a callable
     - ``f(value: str, measure_info: dict) -> str | None``, applied per value.
       Returning ``None`` drops that value and its timestamps. Use this to keep a controlled
       alarm vocabulary while dropping a free-text note measure.

.. warning::

   If you *do* opt into ``"redact"`` or ``"skip"`` for a measure, every **event query** and every
   event-anchored :ref:`DatasetDefinition <event_anchored_regions>` built against it fails on the
   destination, because the values they pair on no longer exist::

       ValueError: from_value 'Anesthesia START' is not in the string vocabulary of measure 4.
       Known values come from get_measure_string_vocabulary().

   If the research copy needs event-anchored cohorts, it needs the vocabulary — leave the default
   ``"transfer"`` in place, or scrub with a callable that keeps the marker values.

.. code-block:: python

   # Nothing is scrubbed unless you say so. This share keeps its controlled vocabularies
   # (alarm names, ventilator modes) and drops one free-text nurse-note measure.
   FREE_TEXT_TAGS = {"clinical_note"}

   def scrub(value, measure_info):
       if measure_info["tag"] in FREE_TEXT_TAGS:
           return None            # drop the value and its timestamps
       return value               # controlled vocabulary: keep verbatim

   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=cohort_def,
       deidentify="patient_mapping_file.csv",
       string_value_policy=scrub,
       time_shift=two_hours_s,
       time_units="s",
   )

Only values inside a transferred time range are ever considered, so the reach of any one policy
is bounded by the cohort's time ranges.

.. note::

   ``string_value_policy`` is a ``transfer_data`` parameter only. The
   :doc:`CLI <cli>` ``export`` command has no equivalent flag, so a CLI export always uses the
   default ``"transfer"`` — string values are copied verbatim, ``--deidentify`` or not. Scrubbing
   a specific measure requires calling :py:func:`transfer_data` from Python.

What de-identification does to the encounter family
***************************************************

When ``deidentify`` is enabled, the encounter family's **patient-level identifier** is
scrambled by default — you do not have to configure anything. When ``deidentify=False``
every field is copied identified and the ``keep_identified`` setting below is a no-op.

The table below is the complete inventory of what de-identification touches on these
tables. IDs are always remapped and times are always shifted (by ``time_shift``) for
referential and temporal integrity — those are not configurable. The **Altered by
de-identification** column lists the values that are changed by default and that
``keep_identified`` can opt back to identified. Text surfaces *outside* the encounter
family are inventoried in the second table below; read both before signing off on a share.

.. list-table:: Encounter-family de-identification (under ``deidentify=True``)
   :header-rows: 1
   :widths: 20 40 40

   * - Table
     - Altered by de-identification
     - Always applied (not configurable)
   * - ``encounter``
     - ``visit_number`` → **scrambled** to a random integer via a consistent per-transfer map
     - ``patient_id`` / ``bed_id`` remapped; ``start_time`` / ``end_time`` / ``last_updated``
       shifted
   * - ``device_encounter``
     - *(nothing)*
     - ``device_id`` / ``encounter_id`` remapped; ``start_time`` / ``end_time`` shifted
   * - ``bed``
     - *(nothing)* — ``name`` transfers **verbatim**
     - ``bed_id`` / ``unit_id`` remapped
   * - ``unit``
     - *(nothing)* — ``name`` transfers **verbatim**
     - ``unit_id`` / ``institution_id`` remapped
   * - ``institution``
     - *(nothing)* — ``name`` transfers **verbatim**
     - ``institution_id`` remapped

``visit_number`` is scrambled because it is a direct visit identifier that joins straight
back to the source record system. Location names are not: "PICU" or "Bed-12" describe where
a recording happened, not whose it is, and the destination needs them to remain meaningful.

The scramble is **stable within a transfer**: the same source ``visit_number`` always maps
to the same integer, so relationships in the data are preserved while the original value is
hidden.

.. _other_free_text_surfaces:

Other text surfaces
********************

The encounter family is not the only place identifying text can live. None of the surfaces below
are altered by ``deidentify``, and none of them are covered by ``keep_identified``. They are
inventoried here so you know exactly what a shared extract contains:

.. list-table:: Text surfaces outside the encounter family (under ``deidentify=True``)
   :header-rows: 1
   :widths: 26 34 40

   * - Surface
     - Default treatment
     - How to control it
   * - **String measure values**
       (alarms, states, notes)
     - Copied **verbatim** — signal content is not a de-identification surface
     - :ref:`string_value_policy <string_value_policy>` — ``transfer`` (default) / ``redact`` /
       ``skip`` / callable, applied only when you ask for it. Timestamps and the interval index
       are preserved in every mode except ``skip``.
   * - **Label names and label text**
     - Copied **verbatim**
     - Label names are user-authored free text (e.g. ``"Artifact by RN Chen"``) and get no
       scrubbing — ``string_value_policy`` does not apply to them. Only a label's device and
       measure IDs are remapped and its times shifted. Pass ``include_labels=False`` to omit
       labels, or curate your label vocabulary before sharing. (A label's *source* is carried
       across by name and re-created in the destination when missing.)
   * - **Device ``tag`` and ``name``**
     - Copied **verbatim**
     - Not pseudonymized. In a hospital these are often room / bay / bed identifiers
       (``OR-1``, ``ICU-3``), which combined with a timeline can be a re-identification vector.
       Rename devices in the destination, or export by patient with device tags you control.
   * - **Bed / unit / institution ``name``**
     - Copied **verbatim**
     - A location name says *where* a recording happened, not *whose* it is, so it is not
       treated as PHI. Rename in the destination if your review requires it.
   * - **Measure ``tag`` / ``units``**
     - Copied **verbatim**
     - Normally desirable; noted here for completeness.
   * - ``log_hl7_adt``
     - **Never** transferred
     - Not configurable.

.. note::

   ``deidentify`` covers patient-level metadata, ``encounter.visit_number`` and time-shifting.
   It does **not** inspect signal values, label names, device metadata, location names, or
   measure metadata. If your compliance review needs those scrubbed, do it before or after the
   transfer — or, for one specific string measure, ask for it explicitly with
   :ref:`string_value_policy <string_value_policy>`.

.. note::

   A label whose ``measure`` is ``None`` — one that annotates a span of a source rather than a
   single signal — transfers unchanged; ``None`` is meaningful and is carried through. A label
   attached to a measure the transfer does **not** include raises a ``ValueError`` naming the
   label and the missing measure, rather than being dropped silently.

Keeping specific fields identified
**********************************

Use the ``keep_identified`` parameter to opt named fields (or whole tables) back to
identified. It is a dictionary of ``{table: [field names]}``, with the shorthand
``"all"`` to keep an entire table identified:

.. code-block:: python

   keep_identified = {
       "encounter": ["visit_number"],  # keep the real visit_number
   }

``encounter.visit_number`` is the only field this currently has an effect on, since it is
the only encounter-family field de-identification alters. Entries for ``bed`` / ``unit`` /
``institution`` names remain valid (they are accepted and validated) but are a no-op: those
names are already identified. Passing an unknown table, or a field that is not a listed
field of that table, raises a ``ValueError``. Omitting ``keep_identified`` (or passing
``{}``) scrambles ``visit_number``, which is the safe default.

.. code-block:: python

   # De-identified export that keeps the real visit numbers, with a 2-hour time shift
   # applied to every timestamp including the encounter times.
   two_hours_s = 2 * 60 * 60
   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=cohort_def,
       deidentify=True,
       keep_identified={"encounter": ["visit_number"]},
       time_shift=two_hours_s,
       time_units="s",
   )

   # ——> Encounters/device_encounters copied for the cohort; bed, unit and institution
   #     names preserved; visit numbers preserved by the opt-in; all times shifted;
   #     patient IDs remapped and patient metadata dropped.

.. note::

   ``keep_identified`` only affects the encounter family. Patient-level de-identification is
   still controlled by ``deidentify`` / ``patient_info_to_transfer`` /
   ``deidentification_functions`` as described above.

.. _to_pandas:

Getting data into pandas
#########################

There is no ``to_dataframe`` / ``to_pandas`` helper on the SDK, the iterator or the ``Window``
object. Building the DataFrame yourself is a few lines: read each measure, then align them with
``merge_asof`` if they are on different clocks.

.. code-block:: python

    import pandas as pd

    # A numeric measure.
    _, hr_t, hr_v = sdk.get_data(hr_id, start_time_n=start_n, end_time_n=end_n, device_id=device_id)
    hr_df = pd.DataFrame({"time_ns": hr_t, "heart_rate": hr_v})

    # A string measure - read with get_string_data, not get_data.
    al_t, al_v = sdk.get_string_data(alarm_id, start_time_n=start_n, end_time_n=end_n,
                                     device_id=device_id)
    al_df = pd.DataFrame({"time_ns": al_t.astype("int64"), "alarm": al_v})

    # Put the text channel next to the numeric one, carrying each alarm forward.
    merged = pd.merge_asof(hr_df.sort_values("time_ns"), al_df.sort_values("time_ns"),
                           on="time_ns", direction="backward")

.. note::

   Read timestamps in **nanoseconds** (the default ``time_units="ns"``) when you care about exact
   times. ``time_units="s"`` returns ``float64`` seconds, and a modern epoch (~1.77e9 s) only has
   about microsecond resolution in a float64 — enough for plotting, not for joining.

.. _export_formats:

Export Formats & CSV Example
############################

AtriumDB supports multiple export formats:

- `"tsc"` (default): Native AtriumDB format (required for use with `AtriumSDK.get_data` and other AtriumSDK waveform data retrieval)
- `"csv"`: Well known, tabular text based format.
- `"npz"`: Numpy arrays
- `"parquet"`: Tabular binary data.
- `"wfdb"`: Binary waveform format.

CSV is generally discouraged for waveform data due to high sample frequency and volume,
disk usage becomes quickly unscalable, but still can be used for small-scale exports if desired.

.. note::

   **String (event / state / text) measures export to** ``csv``, ``npz`` **and** ``parquet``
   as well as to the native ``tsc`` format. Those three have a text-capable value column, so
   what is written is the **decoded strings** — not the ``int64`` dictionary codes, which only
   mean anything alongside the dataset that assigned them.

   ``wfdb`` genuinely cannot hold text. A string measure requested for a ``wfdb`` export (or any
   measure under ``string_value_policy="skip"``) is **not** silently dropped: ``transfer_data``
   raises a ``UserWarning`` naming each omitted measure and the reason, and strikes it from the
   ``meta/definition.yaml`` written into the bundle, so the manifest never claims data the bundle
   does not contain. Share a ``tsc``, ``csv``, ``npz`` or ``parquet`` export instead when a
   collaborator needs your alarm / event text.

.. code-block:: python

   transfer_data(
       src_sdk=main_sdk,
       dest_sdk=export_sdk,
       definition=full_def,
       export_format="csv",
       reencode_waveforms=False
   )

   # ——> Waveforms saved as CSV files under `/export/path/csv/`

The CSV bundle layout is::

    <export_path>/meta/index.db
    <export_path>/meta/definition.yaml
    <export_path>/csv/<device_tag>/<measure_tag>^<freq>Hz^<units>/<ISO-8601 start>.csv

with each file carrying the header ``Timestamp (ns),<measure_tag>``. ``npz`` and ``parquet`` use
the same directory scheme with their own extensions.

.. warning::

   **A CSV export does not round-trip nanosecond timestamps exactly (known defect).** The CSV
   writer divides the ``int64`` nanosecond times by the unit conversion factor before writing —
   which for the default ``ns`` format is a division by 1, and still converts the array to
   ``float64``. A modern nanosecond epoch is around ``1.77e18``, well past the ``2**53`` integer
   range a ``float64`` can represent exactly, so the values written are rounded to roughly the
   nearest few hundred nanoseconds and the last digits are lost::

       stored:  1770000000000000001
       in CSV:  1.77e+18            ->  1770000000000000000

   Sample **values** are unaffected, and the error is far below one sample period for any
   ordinary waveform rate — but a CSV export is not a lossless archive, and a round-trip back
   into AtriumDB will not reproduce the original timestamps bit for bit. Use the native ``tsc``
   format (or ``npz`` / ``parquet``, which keep the ``int64`` column) when exact times matter.

