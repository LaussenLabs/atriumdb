# Atriumdb SDK C Library

This is the C library that supports the encoding and decoding functionatlity of the python atriumdb SDK.

## How to Build
### Linux

The library is built using Cmake, supported to compile using Linux gcc or Windows MinGW.

You can build a new release using the command:

```shell
$ cd Minimal_SDK
$ cmake -Bcmake-build-release -H.
$ cmake --build cmake-build-release --target Block
```

You can clean the project build files using:

```shell
$ cmake --build cmake-build-release --target clean
```

### Docker
If you build using docker it will cross compile both for Linux and Windows.

First you build the docker image using the command:
```shell
$ docker build -t c-build .
```
Then to build the binaries for release from a windows host you use the command:
```shell
$ docker run --name c-build -v "%cd%":/minimal_sdk -w /minimal_sdk --init  -it c-build ./build_release.sh
```
If you want to build the binaries in debug mode use the command:
```shell
$ docker run --name c-build -v "%cd%":/minimal_sdk -w /minimal_sdk --init  -it c-build ./build_debug.sh
```
NOTES: 
- If you are using a linux host replace "%cd%" with $(pwd). 
- Also if you would prefer to run build commands yourself just remove ./build_release.sh from the docker run command 
and it will give you a shell.

## How to Test

Unit tests have been written using Google Test.

You can run the tests by using the commands:

```shell
$ cmake -Bcmake-build-debug -H.
$ cmake --build cmake-build-debug --target Google_Tests_run
$ ./cmake-build-debug/Google_tests/Google_Tests_run
```


## High Level Functionality

The main module of the C Library is Block.

It has 3 public functions:

```C
size_t block_get_buffer_size(const void *time_data, uint64_t num_blocks, const uint64_t *t_block_start,
uint64_t *byte_start, block_metadata_t *headers);

size_t encode_blocks(const void *time_data, const void *value_data, void *encoded_bytes, uint64_t num_blocks,
const uint64_t *t_block_start, const uint64_t *v_block_start, uint64_t *byte_start,
block_metadata_t *headers, const block_options_t *options, uint16_t num_threads);

void decode_blocks(void *time_data, void *value_data, void *encoded_bytes, uint64_t num_blocks,
const uint64_t *t_block_start, const uint64_t *v_block_start, const uint64_t *byte_start,
const uint64_t *t_byte_start, uint16_t num_threads);
```

`block_get_buffer_size` is used to calculate the number of bytes to be allocated for the `encoded_bytes` parameter in 
encode_blocks.

`encode_blocks` fills the buffer in encoded_bytes with the encoded blocks and returns the total number of encoded bytes.

`decode_blocks` decodes the block data present in `encoded bytes` and places the values in `time_data` and `value_data` 
which should already be appropriately sized and allocated by the user.

## Modules

There are a number of modules that work together to implement the functionality of Block.

- `Block_Header` - Contains a header file defining the block header using a C struct.
- `Compression` - Implements a compression agnostic `my_compress` and `my_decompress`, calling 3rd party compression 
  libraries in the backend such as lz4 and zstd.
- `Delta` - Performs the delta encoding of integer values and depends on the `Entropy` module to calculate the order of 
  delta encoding with the lowest Shannon Entropy, which suggests which order will compress the smallest.
- `Distinct` - Assists the Entropy module by counting the number of distinct elements in an array.
- `Entropy` - Assists the Delta module by calculating the Shannon entropy of a given array.
- `Freq_Period_Converter` - Assists the Time module by converting a frequency in nanohertz to a period in nanoseconds 
  and vice versa.
- `Gap` - Assists the Time module by converting an array of timestamps into an array of gap indices/durations and 
  vice versa.
- `Size` - Assists the Value module by calculating the smallest bit depth allowable for a given array, and converting 
  between various bit-depths.
- `Time` - Assists the Block module by handling the encoding/decoding of the time information of a block.
- `Value` - Assists the Block module by handling the encoding/decoding of the value information of a block.
