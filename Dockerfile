# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "debconf debconf/frontend select Noninteractive" | \
        debconf-set-selections

RUN   apt-get update -y -q && \
      apt-get install -y -q --no-install-recommends \
          apt-transport-https \
          ca-certificates \
          gnupg \
          lsb-release \
          wget && \
      code_name=$(lsb_release --codename --short) && \
    apt-get update -y -q && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Installs C++ toolchain and dependencies
RUN apt-get update -y -q && \
    apt-get install -y -q --no-install-recommends \
        autoconf \
        ca-certificates \
        ccache \
        cmake \
        curl \
        g++ \
        gcc \
        gdb \
        git \
        libbenchmark-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libbrotli-dev \
        libbz2-dev \
        libc-ares-dev \
        libcurl4-openssl-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        liblz4-dev \
        libprotobuf-dev \
        libprotoc-dev \
        libradospp-dev \
        libre2-dev \
        libsnappy-dev \
        libssl-dev \
        libthrift-dev \
        libutf8proc-dev \
        libzstd-dev \
        make \
        ninja-build \
        nlohmann-json3-dev \
        pkg-config \
        protobuf-compiler \
        python3-dev \
        python3-pip \
        python3-rados \
        rados-objclass-dev \
        rapidjson-dev \
        rsync \
        tzdata \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

COPY ci/scripts/install_minio.sh /arrow/ci/scripts/
RUN /arrow/ci/scripts/install_minio.sh latest /usr/local

COPY ci/scripts/install_gcs_testbench.sh /arrow/ci/scripts/
RUN /arrow/ci/scripts/install_gcs_testbench.sh default

COPY ci/scripts/install_ceph.sh /arrow/ci/scripts/
RUN /arrow/ci/scripts/install_ceph.sh

COPY ci/scripts/install_sccache.sh /arrow/ci/scripts/
RUN /arrow/ci/scripts/install_sccache.sh unknown-linux-musl /usr/local/bin

# The following dependencies will be downloaded due to missing/invalid packages
# provided by the distribution:
# - Abseil is not packaged
# - libc-ares-dev does not install CMake config files
# - flatbuffer is not packaged
# - libgtest-dev only provide sources
# - libprotobuf-dev only provide sources
# ARROW-17051: this build uses static Protobuf, so we must also use
# static Arrow to run Flight/Flight SQL tests
ENV absl_SOURCE=BUNDLED \
    ARROW_CUDA=ON \
    ARROW_BUILD_STATIC=OFF \
    ARROW_BUILD_TESTS=OFF \
    ARROW_DEPENDENCY_SOURCE=SYSTEM \
    ARROW_SUBSTRAIT=OFF \
    ARROW_DATASET=ON \
    ARROW_FLIGHT=OFF \
    ARROW_GANDIVA=OFF \
    ARROW_GCS=OFF \
    ARROW_HDFS=OFF \
    ARROW_HOME=/usr/local \
    ARROW_INSTALL_NAME_RPATH=OFF \
    ARROW_NO_DEPRECATED_API=ON \
    ARROW_ORC=OFF \
    ARROW_PARQUET=ON \
    ARROW_PLASMA=ON \
    ARROW_S3=OFF \
    ARROW_USE_ASAN=OFF \
    ARROW_USE_CCACHE=ON \
    ARROW_USE_UBSAN=OFF \
    ARROW_WITH_BROTLI=OFF \
    ARROW_WITH_BZ2=OFF \
    ARROW_WITH_LZ4=OFF \
    ARROW_WITH_OPENTELEMETRY=OFF \
    ARROW_WITH_SNAPPY=OFF \
    ARROW_WITH_ZLIB=OFF \
    ARROW_BUILD_STATIC=OFF \
    ARROW_BUILD_TESTS=OFF \
    ARROW_BUILD_UTILITIES=OFF \
    ARROW_COMPUTE=ON \
    ARROW_CSV=ON \
    ARROW_FILESYSTEM=ON \
    ARROW_JSON=ON \
    ARROW_USE_GLOG=OFF \
    ARROW_WITH_ZSTD=OFF \
    ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-${llvm}/bin/llvm-symbolizer \
    AWSSDK_SOURCE=BUNDLED \
    google_cloud_cpp_storage_SOURCE=BUNDLED \
    gRPC_SOURCE=BUNDLED \
    GTest_SOURCE=BUNDLED \
    ORC_SOURCE=BUNDLED \
    PARQUET_BUILD_EXAMPLES=OFF \
    PARQUET_BUILD_EXECUTABLES=OFF \
    Protobuf_SOURCE=BUNDLED \
    PATH=/usr/lib/ccache/:$PATH \
    PYTHON=python3 \
    xsimd_SOURCE=BUNDLED

COPY . /arrow
RUN /bin/bash -c "/arrow/ci/scripts/cpp_build.sh /arrow /build" && rm /build -rf
RUN /bin/bash -c "/arrow/ci/scripts/integration_skyhook.sh /build"

RUN apt-get update -y -q && \
    apt-get install -y -q \
        python3 \
        python3-pip \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install -U pip setuptools wheel

COPY python/requirements-build.txt \
     python/requirements-test.txt \
     /arrow/python/

RUN pip install \
    -r arrow/python/requirements-build.txt

RUN /bin/bash -c "/arrow/ci/scripts/python_build.sh /arrow /build"
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 
RUN cd /arrow/python/pyarrow/torch-extension/tensorstore/ && python setup.py install && pip3 install timm cuda-python

CMD /bin/bash -c "sleep infinity"
