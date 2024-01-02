Dataset Iterators
========================

Iterating Over a Dataset
------------------------

Often we are interested in working with relatively small windows of data at a time. For visualizing or pre-processing
small amounts of data at once, or when we are training a model.
However the primary way of querying data, `AtriumSDK.get_data  <contents.html#atriumdb.AtriumSDK.get_data>`_ incurs an
overhead cost everytime it is called. This makes it an inefficient means of collecting a large amount of small windows
of data.

For this reason, AtriumDB has a `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ method, that
preloads large amounts of data in your RAM, and feeds it to you piece by piece in an iterable Class.

`AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ also does the job of windowing and indexing
your data for you, which makes tasks like training a model much simpler.

What is the ``get_iterator`` method?
###################################################

The `AtriumSDK.get_iterator  <contents.html#atriumdb.AtriumSDK.get_iterator>`_ method allows users to define a set of
measures and specific patients or devices over particular time intervals through the
`DatasetDefinition Class <contents.html#atriumdb.DatasetDefinition>`_ or a filename pointing to a valid
:ref:`definition_file_format`. It ensures that the defined cohort exists within the available dataset.
If portions of the cohort definition fall outside the dataset's boundaries, the method trims the cohort to fit within
the available dataset and raises warnings about any data that has been omitted.

The method returns a ``DatasetIterator`` object. This object implements ``__len__``, ``__next__`` and ``__getitem__``
methods, allowing it to be looped over or iterated upon using traditional python code conventions like
``for window in iterator`` or ``window = next(iterator)``. It also allows the class to directly facilitate a
PyTorch Dataloader.

How to Use
#################

1. Initialize your SDK instance:

.. code-block:: python

   sdk = AtriumSDK(dataset_location=local_dataset_location)

2. Define the measures and patient or device cohorts:

**Example 1: By Patient**

.. code-block:: python

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

**gap_tolerance** informs the auto-generated definition intervals when using "all" mode, defining the desired largest
possible section of no data to be emitted by the windows.

**time_units** defines the time units of ``window_duration``, ``window_slide`` and ``gap_tolerance`` options are
``["s", "ms", "us", "ns"]``, default ``"ns"``.

.. code-block:: python

   num_windows_prefetch = 100_000  # preload 100,000 windows before emitting
   gap_tolerance = 3600_000_000_000  # No gaps between data less than an hour. (NaNs will fill the gaps)
   time_units = "ns"

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
        for (measure_tag, measure_freq_nhz, measure_units), signal_dict in window.signals.items():
            print(measure_tag, measure_freq_nhz, measure_units, signal_dict['measure_id'])
            print('times', signal_dict['times'])
            print('values', signal_dict['values'])
            print('expected_count', signal_dict['expected_count'])
            print('actual_count', signal_dict['actual_count'])

        # Total Data Matrix, useful for feeding a model.
        print(iterator.get_array_matrix(window_i))

You can find explanations of the returned Window object in the :ref:`window_format` section below.

.. _window_format:

Window Format
#####################

The ``Window`` class represents a data structure for windowed data output by the
`DatasetIterator Class <contents.html#atriumdb.DatasetIterator>`_, it includes the raw
data organized into signal dictionaries, along with associated metadata.

**Attributes**:

- ``signals`` : ``dict``
    A dictionary where each entry corresponds to a different measure signal, making it easier to handle measures of different frequencies. The keys of this dictionary are tuples, each consisting of the measure tag, the frequency of the measure (in nHz), and the units of the measure. The values are dictionaries containing metadata and data for each measure.

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
    An identifier representing the device from which the data was captured. It will be ``None`` if the data was retrieved based on a patient ID.

- ``patient_id`` : ``int``
    An identifier representing the patient associated with the data. It will be ``None`` if the data was retrieved based on a device ID.

Example of the ``signals`` dictionary:

.. code-block:: python

    {
        ('heart_rate', 1_000_000_000, 'bpm'): {
            'times': np.array([1, 2, 3, ...]),
            'values': np.array([70, 71, 69, ...]),
            'expected_count': 100,
            'actual_count': 100,
            'measure_id': 123,
        },
        ('temperature', 10_000_000, 'C'): {
            'times': np.array([0, 10, 20, ...]),
            'values': np.array([36.6, 36.7, np.nan, ...]),
            'expected_count': 10,
            'actual_count': 9,
            'measure_id': 456,
        }
    }

.. _definition_file_format:

Definition File Format
------------------------------

Source Types
#################

The YAML file defines various source types (``patient_ids``, ``mrns``, ``device_ids``, and ``device_tags``). For each source type, specific sources or entities are identified by unique names.

Time Entries
#################

For each source/entity, you can provide multiple time entries. Each time entry describes a relevant time period or event for that source. There are three types of time specifications:

1. **Interval-based**: This type specifies a continuous interval with a ``start`` and/or ``end`` time.

   - ``start``: The beginning of the interval (nanosecond Unix Epoch Time).
   - ``end``: The end of the interval (nanosecond Unix Epoch Time).

2. **Event-based**: This type specifies an event time and the time period before and after the event.

   - ``time0``: The exact time of the event (nanosecond Unix Epoch Time).
   - ``pre``: Duration before the event (in nanoseconds).
   - ``post``: Duration after the event (in nanoseconds).

3. **All** All available time data can be specified using the ``all`` keyword.

Measures
#################

The ``measures`` section lists various measures to be considered. Each measure can either be:

1. The measure tag, if there is only one measure with that tag.
2. A complete measure triplet which includes:

   - ``tag``: The tag identifying the measure.
   - ``freq_hz`` or ``freq_nhz``: The frequency of the measure in Hertz (floating) or nanoHertz (integer).
   - ``units``: The unit of the measure (e.g., volts, bpm).

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


Definition YAML Examples
-----------------------------

Creating a DatasetDefinition object
###################################

You can create a `DatasetDefinition <contents.html#atriumdb.DatasetDefinition>`_ object in several ways:

1. Reading from an existing YAML file:

   .. code-block:: python

      dataset_definition = DatasetDefinition(filename="/path/to/my_definition.yaml")

2. Creating an empty definition:

   .. code-block:: python

      dataset_definition = DatasetDefinition()

3. Creating a definition with measures and no regions:

   .. code-block:: python

      measures = ["measure_tag_1", ("measure_tag_2", 62.5, "measure_units_2")]
      dataset_definition = DatasetDefinition(measures=measures)

4. Creating a definition with measures and regions:

   .. code-block:: python

      device_tags = {"tag_1": [{'start': start_time_nano_1, 'end': end_time_nano_1}], "tag_2": [{'time0': event_time_nano_2, 'pre': nano_before_event_2, 'post': nano_after_event_2}]}
      dataset_definition = DatasetDefinition(measures=measures, device_tags=device_tags)


Adding to a DatasetDefinition object
####################################

1. Adding a measure:

   You can add a measure by its tag if there is only one measure with that tag. If there are multiple measures with the same tag, you need to specify the frequency and units as well.

   .. code-block:: python

      sdk.insert_measure(measure_tag="ART_BLD_PRESS", freq=62.5, units="mmHG", freq_units="Hz")
      dataset_definition.add_measure(tag="ART_BLD_PRESS")  # Okay

      sdk.insert_measure(measure_tag="ART_BLD_PRESS", freq=250, units="mmHG", freq_units="Hz")
      dataset_definition.add_measure(tag="ART_BLD_PRESS")  # ValueError: More than 1 measure has that tag
      >>> ValueError
      dataset_definition = DatasetDefinition()
      dataset_definition.add_measure(measure_tag="ART_BLD_PRESS", freq=62.5, units="mmHG")  # Okay
      dataset_definition.add_measure(measure_tag="ART_BLD_PRESS", freq=250, units="mmHG")  # Okay

2. Adding a region:

   You can add a region by specifying a ``device_tag``, ``patient_id``, or ``mrn``, along with the relevant time parameters. Only one of ``patient_id``, ``mrn``, ``device_id``, or ``device_tag`` should be specified.

   .. code-block:: python

      dataset_definition.add_region(device_tag="tag_1", start=1693499415_000_000_000, end=1693583415_000_000_000)
      dataset_definition.add_region(patient_id=12345, start=1693364515_000_000_000, end=1693464515_000_000_000)
      dataset_definition.add_region(mrn=1234567, start=1659344515_000_000_000, end=1660344515_000_000_000)
      dataset_definition.add_region(mrn="7654321", time0=1659393745_000_000_000, pre=3600_000_000_000, post=3600_000_000_000)

Saving a DatasetDefinition object
#################################

Once you have defined all the measures and regions, you can save the definition to a YAML file.

.. code-block:: python

   dataset_definition.save(filepath="path/to/saved/definition.yaml")

Note that the file extension must be ``.yaml`` or ``.yml``.

If you would like to overwrite an existing file, include the ``force=True`` keyword parameter.