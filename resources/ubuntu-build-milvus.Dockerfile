ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

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
        # Provide CUDA dependencies by libnvidia-compute*
        libnvidia-compute-${NV_DRIVER_VERSION} \
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
	libssl-dev zlib1g-dev libboost-regex-dev libboost-program-options-dev \
	libboost-system-dev libboost-filesystem-dev \
	libboost-serialization-dev python3-dev libboost-python-dev \
	libcurl4-openssl-dev gfortran libtbb-dev libzstd-dev libaio-dev \
	uuid-dev libpulse-dev \
    # Remove cuda-compat* from nvidia/cuda:x86_64 images, provide CUDA dependencies by libnvidia-compute* instead
    && apt-get remove -y openjdk-11-* cuda-compat* || apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV GOLANG_VERSION 1.21.12
RUN wget https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-amd64.tar.gz \
    && rm go${GOLANG_VERSION}.linux-amd64.tar.gz

ENV PATH=/usr/local/go/bin:$PATH
