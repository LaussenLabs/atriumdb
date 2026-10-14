//
// Created by Will Dixon on 2021-05-13.
//
#include "gtest/gtest.h"

extern "C" {
#include <compression.h>
#include <stdint.h>
}

TEST(CompressionTests, ShortTest){
    /* Create some data to test compression */
    int64_t mockData[] = {50, 1000000};
    int numValues = sizeof(mockData) / sizeof(mockData[0]);

    /* Use the most common setting LZ4 High Compression */
    uint8_t compressionType = 5, compressionLevel = 12;

    /* allocate some memory for the compressed bytes */
    size_t compressionBufferSize = get_compress_buffer_size(sizeof(mockData), compressionType);
    unsigned char * compressionBuffer = (unsigned char *)malloc(compressionBufferSize);

    /* Perform Compression */
    size_t compressedSize = my_compress((unsigned char *)mockData, compressionBuffer, sizeof(mockData),
                                        compressionBufferSize, compressionType, (int8_t) compressionLevel);

    /* Array to store decompressed result initialized with zeros */
    int64_t decompressResult[2] = {0, 0};

    size_t bytesRead = my_decompress(compressionBuffer, (unsigned char *)decompressResult,
                                      compressedSize, sizeof(mockData), compressionType);

    ASSERT_EQ(bytesRead, sizeof(mockData));

    int i;
    for(i=0; i < numValues; i++){
        EXPECT_EQ(mockData[i], decompressResult[i]) << i;
    }
    free(compressionBuffer);
}
