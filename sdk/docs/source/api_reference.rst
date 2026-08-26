.. _api_reference:

API Reference
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. Every class and function below is documented exactly ONCE. The curated
   ``autoclass``/``automethod`` listing that follows groups the API by topic, so
   ``automodule`` must not document those same objects a second time: a docstring
   rendered twice defines each of its ``.. _..._label:`` targets twice, docutils
   discards duplicate explicit targets, and every ``:ref:`` to them then fails with
   "undefined label". The exclusions below leave ``automodule`` responsible only for
   the module-level members the curated list does not cover -- the gap-array helpers,
   the ``T_TYPE_*``/``V_TYPE_*`` constants, ``WindowConfig`` and ``UNKNOWN_STRING_CODE``.

.. automodule:: atriumdb
   :members:
   :exclude-members: AtriumSDK, DatasetDefinition, DatasetIterator, Window,
      partition_dataset, combine_definitions, cross_validate_dataset, transfer_data

.. autoclass:: atriumdb.AtriumSDK

   .. automethod:: __init__
   .. automethod:: create_dataset

   .. automethod:: get_data
   .. automethod:: get_string_data
   .. automethod:: get_headers
   .. automethod:: write_data_easy
   .. automethod:: write_data
   .. automethod:: write_buffer
   .. automethod:: write_segment
   .. automethod:: write_segments
   .. automethod:: write_time_value_pairs

   .. automethod:: load_device
   .. automethod:: load_definition

   .. automethod:: get_measure_id
   .. automethod:: get_measure_info
   .. automethod:: update_measure
   .. automethod:: get_measure_id_list_from_tag
   .. automethod:: search_measures
   .. automethod:: get_all_measures
   .. automethod:: insert_measure
   .. automethod:: get_or_insert_measure

   .. automethod:: get_device_id
   .. automethod:: get_device_info
   .. automethod:: search_devices
   .. automethod:: get_all_devices
   .. automethod:: insert_device
   .. automethod:: get_or_insert_device

   .. automethod:: insert_patient
   .. automethod:: get_patient_info
   .. automethod:: get_all_patients
   .. automethod:: get_mrn_to_patient_id_map
   .. automethod:: get_patient_id_to_mrn_map
   .. automethod:: get_patient_id
   .. automethod:: get_mrn

   .. automethod:: get_device_patient_data
   .. automethod:: insert_device_patient_data
   .. automethod:: convert_patient_to_device_id
   .. automethod:: convert_device_to_patient_id

   .. automethod:: get_labels
   .. automethod:: insert_label
   .. automethod:: insert_labels
   .. automethod:: delete_labels
   .. automethod:: get_label_name_id
   .. automethod:: get_label_name_info
   .. automethod:: get_all_label_names
   .. automethod:: get_label_name_children
   .. automethod:: get_label_name_parent
   .. automethod:: insert_label_name
   .. automethod:: get_all_label_name_descendents

   .. automethod:: get_label_source_id
   .. automethod:: get_label_source_info
   .. automethod:: get_all_label_sources
   .. automethod:: insert_label_source
   .. automethod:: get_label_time_series

   .. automethod:: get_measure_string_vocabulary
   .. automethod:: get_string_values_present
   .. automethod:: get_event_intervals

   .. automethod:: insert_encounter
   .. automethod:: get_encounters
   .. automethod:: get_device_patient_encounters

   .. automethod:: get_iterator
   .. automethod:: get_interval_array

.. autoclass:: atriumdb.intervals.Intervals

   .. automethod:: __init__
   .. automethod:: intersection
   .. automethod:: difference
   .. automethod:: union
   .. automethod:: duration
   .. automethod:: is_empty
   .. automethod:: contains
   .. automethod:: gaps

.. autoclass:: atriumdb.DatasetDefinition

   .. automethod:: __init__
   .. automethod:: build_from_intervals
   .. automethod:: validate
   .. automethod:: filter
   .. automethod:: add_measure
   .. automethod:: add_label
   .. automethod:: add_region
   .. automethod:: combine
   .. automethod:: save

.. autoclass:: atriumdb.DatasetIterator

   .. automethod:: __next__
   .. automethod:: __iter__
   .. automethod:: decode_window_strings

.. _window_class:

.. autoclass:: atriumdb.windowing.window.Window

   .. automethod:: decode_string_signal


.. autofunction:: partition_dataset
.. autofunction:: combine_definitions
.. autofunction:: cross_validate_dataset
.. autofunction:: transfer_data


Index
--------------------

* :ref:`genindex`
