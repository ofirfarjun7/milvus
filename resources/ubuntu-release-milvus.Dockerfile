ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# FROM rdmz-harbor.rdmz.labs.mlnx/ucx/x86_64/ubuntu22.04/builder:mofed-5.7-0.2.3.0

ARG NV_DRIVER_VERSION
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y \
	clang-format \
	clang-tidy \
	gcc \
	vim \
	sudo \
	cmake \
	ninja-build \
	software-properties-common \
	apt-file \
	automake \
	default-jdk \
	dh-make \
	g++ \
	git \
	openjdk-8-jdk \
	libcap2 \
	libnuma-dev \
	libtool \
	make \
	maven \
	pkg-config \
	udev \
	wget \
	environment-modules \
	python3 \
	python3-pip \
	python3-dev \
	python3.10-venv \
	build-essential \
	libglib2.0-0 \
	iproute2 \
	libglvnd-dev \
	libgl-dev \
	ethtool graphviz pciutils lsof libusb-1.0-0 bison tk kmod libfuse2 swig libpci3 chrpath flex tcl \
	openssl \
	locales \
	gdb \
	curl \
	zip \
	unzip \
	tar \
	libopenblas-dev \
	xsltproc \
	tmux \
	libboost-all-dev \
	libmbedtls-dev \
	lcov \
	m4 \
	autoconf \
	ccache \
	libboost-system-dev libboost-filesystem-dev \
	libboost-serialization-dev python3-dev libboost-python-dev \
	libcurl4-openssl-dev gfortran libtbb-dev libzstd-dev libaio-dev \
	uuid-dev libpulse-dev \
	libssl-dev zlib1g-dev libboost-regex-dev libboost-program-options-dev \
    && apt-get remove -y openjdk-11-* cuda-compat* || apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG MOFED_VERSION=23.10-3.2.2.0
ARG MOFED_OS=ubuntu22.04
ARG ARCH=x86_64
ENV MOFED_DIR MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-${ARCH}
ENV MOFED_IMAGE ${MOFED_DIR}.tgz
RUN wget --no-verbose http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/${MOFED_IMAGE} && \
	tar -xzf ${MOFED_IMAGE}
RUN ${MOFED_DIR}/mlnxofedinstall --all -q \
        --user-space-only \
        --without-fw-update \
        --skip-distro-check \
        --without-ucx \
        --without-hcoll \
        --without-openmpi \
        --without-sharp \
        --force && \
    rm -rf ${MOFED_DIR} && rm -rf *.tgz

run mkdir -p /Devl
COPY ucx-ubuntu /Devl/ucx
WORKDIR /milvus
COPY openssl/openssl-1.1.1q /milvus/openssl-1.1.1q
COPY cmake-3.26.4 /milvus/cmake-3.26.4
COPY go1.21.12 /milvus/go1.21.12
COPY milvus/bin /milvus/bin
COPY milvus/configs /milvus/configs
COPY milvus/internal/core/output/lib /milvus/lib

ENV CPATH /usr/local/cuda/include:${CPATH}
ENV PATH=/hpc/newhome/ofarjon/vdb/cmake-3.26.4/bin:/hpc/newhome/ofarjon/vdb/go1.21.12/go/bin:/milvus/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/milvus/lib:/milvus/openssl-1.1.1q:/Devl/ucx/install-ubuntu/lib:/usr/share:/usr/lib:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64:${LIBRARY_PATH}
