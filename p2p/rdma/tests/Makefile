# SPDX-License-Identifier: (GPL-2.0 OR BSD-2-Clause)

# !!! Remember to conda deactivate before running make !!!

# Platform selection: set USE_HIP=1 for ROCm/HIP, default is CUDA
USE_HIP ?= 0

CUDA_HOME := /usr/local/cuda
HIP_HOME := /opt/rocm
DIETGPU_ROOT := ../../../thirdparty/dietgpu
DIETGPU_FLOAT_DIR := $(DIETGPU_ROOT)/dietgpu/float

# Common flags
COMMON_FLAGS := -I ../ -I ../providers -I ../../util -I ../../nccl \
	-I /opt/amazon/efa/include -L /opt/amazon/efa/lib -libverbs -lefa -lpthread \
	-I ../../../include -I ../../../collective/rdma -I $(DIETGPU_ROOT) -std=c++17 -lgflags

ifeq ($(USE_HIP),1)
	# HIP/ROCm flags
	CXXFLAGS = -D__HIP_PLATFORM_AMD__ \
		-I$(HIP_HOME)/include -L$(HIP_HOME)/lib -lamdhip64 \
		-L$(DIETGPU_FLOAT_DIR) -ldietgpu_float \
		-Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) \
		$(COMMON_FLAGS)
else
	# CUDA flags
	CXXFLAGS = -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -lcuda \
		-L$(DIETGPU_FLOAT_DIR) -ldietgpu_float \
		-Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) \
		$(COMMON_FLAGS)
endif

all: test_efa_endpoint test_compression

test_efa_endpoint: test_efa_endpoint.cpp ../../util/common.h ../rdma_device.h ../rdma_context.h ../efa_channel.h ../efa_endpoint.h ../../util/epoll_client.h ../../util/epoll_server.h ../../../collective/rdma/param.cc
	g++ -O3 -g test_efa_endpoint.cpp ../../../collective/rdma/param.cc -o test_efa_endpoint $(CXXFLAGS)

test_compression: test_compression.cpp ../compression.h ../../util/common.h ../../util/memory_allocator.h
	g++ -O3 -g test_compression.cpp -o test_compression $(CXXFLAGS)

clean:
	rm -f test_efa_endpoint test_efa_endpoint.o test_compression test_compression.o \
		test_compressed_ack test_async_decompress async_decode_gate.o

# CPU protocol check and a standalone test of the production async decoder.
# The GPU target needs CUDA and the bundled DietGPU library, with no Python,
# MR registration or network initialization.
test_compressed_ack: test_compressed_ack.cpp ../compressed_ack.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -I.. $< -o $@

test_async_decompress: test_async_decompress.cpp ../dietgpu_decompress.cc ../dietgpu_decompress.h async_decode_gate.o
	$(CXX) -std=c++17 -O2 -DUSE_DIETGPU -I.. \
		-I../../../include -I$(DIETGPU_ROOT) -I$(CUDA_HOME)/include \
		$< ../dietgpu_decompress.cc async_decode_gate.o -L$(CUDA_HOME)/lib64 -lcudart -lcuda \
		-L$(DIETGPU_FLOAT_DIR) -ldietgpu_float -lpthread \
		-Wl,-rpath,$(abspath $(DIETGPU_FLOAT_DIR)) -o $@

async_decode_gate.o: async_decode_gate.cu async_decode_gate.h
	$(CUDA_HOME)/bin/nvcc -std=c++17 -O2 -arch=sm_90 -c $< -o $@
