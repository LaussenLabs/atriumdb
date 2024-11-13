.. atriumdb documentation master file, created by
   sphinx-quickstart on Wed Jan 25 14:44:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API Reference
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: atriumdb
   :members:
   :special-members:

.. autoclass:: atriumdb.AtriumSDK

   .. automethod:: __init__
   .. automethod:: create_dataset

   .. automethod:: get_data
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
   .. automethod:: search_measures
   .. automethod:: get_all_measures
   .. automethod:: insert_measure

   .. automethod:: get_device_id
   .. automethod:: get_device_info
   .. automethod:: search_devices
   .. automethod:: get_all_devices
   .. automethod:: insert_device

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
   .. automethod:: insert_label_source

   .. automethod:: get_iterator
   .. automethod:: get_interval_array

.. autoclass:: atriumdb.DatasetDefinition

   .. automethod:: __init__
   .. automethod:: add_measure
   .. automethod:: add_label
   .. automethod:: add_region
   .. automethod:: save

.. autoclass:: atriumdb.DatasetIterator

   .. automethod:: __next__
   .. automethod:: __iter__


Index
--------------------

* :ref:`genindex`
