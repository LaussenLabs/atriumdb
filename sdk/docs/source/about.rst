About
------------

.. toctree::
   :maxdepth: 2

AtriumDB is a comprehensive solution for the management and analysis of physiological waveform data. It includes a powerful SDK for data compression, storage, and retrieval and a RESTful API for remote access and a WAL System which enables continuous ingestion of streaming data, making it ideal for storing data from clinical systems in real-time.

The AtriumDB SDK is a powerful and versatile python library, designed to meet the needs of data scientists and organizations dealing with large amounts of physiological waveform data. This library combines the computational efficiency of a C library with the convenience and familiarity of a Python interface, making it easy for data scientists to use. The SDK provides a comprehensive solution for data compression, storage, and retrieval, with the ability to index data by signal type, data source, and time, providing efficient and reliable data access. Additionally, the SDK incorporates the core functionality of the API, enabling secure remote access to data through Auth0 authentication.

The AtriumDB API is a secure and reliable solution for remote access to AtriumDB datasets. Designed as a RESTful interface, the API provides a flexible and scalable way to access and manipulate data stored in AtriumDB. The API leverages the powerful data compression capabilities of the SDK, reducing the amount of data that needs to be transmitted over the network connection and improving overall efficiency and performance. All remote connections are authenticated by Auth0, ensuring a high standard of data security.

The WAL System is a sophisticated and efficient tool designed to manage large amounts of streaming data. It uses Write Append Log (WAL) files to accumulate incoming data, allowing for seamless and continuous operation. After a predetermined interval, the WAL files are closed and new ones are opened, ensuring the data is efficiently managed and stored. The closed files are then processed and ingested into AtriumDB via the SDK.