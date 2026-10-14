# dockerfile for making a build container
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    tzdata \
    cmake \
    build-essential \
    gdb \
    libzstd-dev \
    liblz4-dev \
    python3.10 \
    mingw-w64 \
    && rm -rf /var/lib/apt/lists/*

ENV TZ="America/Toronto"

COPY "/build_release.sh" .
COPY "/build_debug.sh" .

# convert sh file line endings from windows to linux
RUN sed -i -e 's/\r$//' /build_release.sh
RUN sed -i -e 's/\r$//' /build_debug.sh

RUN chmod 755 /build_release.sh
RUN chmod 755 /build_debug.sh

CMD ["/bin/bash"]