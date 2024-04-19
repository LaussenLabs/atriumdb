# dockerfile for making a build container
FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    tzdata \
    cmake \
    build-essential \
    gdb \
    python3 \
    mingw-w64 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the environment timezone
ENV TZ="America/Toronto"
ENV MINGW_HOME=/usr

# Install libzstd from source
RUN wget -qO- https://github.com/facebook/zstd/releases/download/v1.4.9/zstd-1.4.9.tar.gz | tar xz && \
    cd zstd-1.4.9 && \
    make && make install && \
    cd .. && rm -rf zstd-1.5.6

# Install liblz4 from source
RUN wget -qO- https://github.com/lz4/lz4/archive/v1.9.3.tar.gz | tar xz && \
    cd lz4-1.9.3 && \
    make && make PREFIX=/usr install && \
    cd .. && rm -rf lz4-1.9.3

# Copy build scripts
COPY "/build_release.sh" .
COPY "/build_debug.sh" .

# convert sh file line endings from windows to linux
RUN sed -i -e 's/\r$//' /build_release.sh
RUN sed -i -e 's/\r$//' /build_debug.sh

# Make scripts executable
RUN chmod 755 /build_release.sh
RUN chmod 755 /build_debug.sh

CMD ["/bin/bash"]