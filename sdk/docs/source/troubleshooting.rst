.. _troubleshooting:

Troubleshooting
=================

This page addresses common issues encountered when using AtriumDB and provides recommendations for resolving them.

**Contents**

- :ref:`OpenMP Interference <openmp_interference>`


.. _openmp_interference:

OpenMP Interference
----------------------

**Problem**

AtriumDB uses OpenMP to facilitate multithreaded encoding and decoding of data blocks.
To control the number of threads OpenMP uses, AtriumDB dynamically sets the environment variable `OMP_NUM_THREADS`
during each read or write operation, based on the `num_threads` parameter provided during `AtriumSDK` initialization.
This behavior can interfere with other libraries that also use OpenMP, such as PyTorch, leading to performance issues or unexpected behavior.

**Solution**

To avoid conflicts, set the `AtriumSDK.num_threads` parameter during initialization to a value suitable for all OpenMP-dependent libraries in your workflow.

For example:

.. code-block:: python

    from atriumdb import AtriumSDK

    sdk = AtriumSDK(dataset_location="./my_dataset", num_threads=4)

**Important Notes:**

- AtriumDB can affect other libraries using OpenMP by dynamically setting `OMP_NUM_THREADS`. Similarly, other libraries that modify this variable can impact AtriumDB’s performance.
- Monitor the overall threading configuration of your application to ensure harmony across all libraries.

**Duplicate OpenMP runtime on macOS**

The macOS wheels include their own OpenMP runtime (libomp). If another package that includes its own copy, such as
PyTorch or scikit-learn, is loaded in the same process, OpenMP can stop the program with
`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`.
Setting the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` before starting Python lets both runtimes load.
The OpenMP runtime documents this as an unsafe workaround, so prefer importing only one of the packages per process.
