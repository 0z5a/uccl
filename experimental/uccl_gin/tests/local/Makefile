CUDA_HOME ?= /usr/local/cuda
SM ?= 120
CXX ?= g++
NUMA_INCLUDE_DIR ?=
NUMA_LIBRARY_DIR ?=
NCCL_INCLUDE_DIR ?=
BUILD_DIR ?= build/sm$(SM)

GIN := ../..
ROOT := ../../../..
NVCC := $(CUDA_HOME)/bin/nvcc
INCLUDES := -I$(GIN) -I$(ROOT)/include -I$(CUDA_HOME)/include $(if $(NUMA_INCLUDE_DIR),-I$(NUMA_INCLUDE_DIR))
HOSTFLAGS := -O3 -std=c++17 -pthread -Wall $(INCLUDES)
DEVFLAGS := -O3 -std=c++17 -arch=sm_$(SM) -ccbin=$(CXX) --expt-relaxed-constexpr $(INCLUDES) -Xcompiler=-pthread,-Wall
DEVFLAGS += $(EXTRA_DEVFLAGS)
LIBS := -L$(CUDA_HOME)/lib64 -Xlinker=-rpath -Xlinker=$(CUDA_HOME)/lib64 $(if $(NUMA_LIBRARY_DIR),-L$(NUMA_LIBRARY_DIR)) -lnuma -lcuda

.PHONY: all clean coop-tests adapter-tests adapter-compile-fail compile-fail
all: $(BUILD_DIR)/capabilities $(BUILD_DIR)/queue_smoke

coop-tests: $(BUILD_DIR)/coop_flush $(BUILD_DIR)/network_gate.o

adapter-tests: $(BUILD_DIR)/adapter_signal

$(BUILD_DIR)/adapter_signal: adapter_signal.cu queue_fixture.hpp $(ROOT)/thirdparty/nccl-ep/adapter/uccl_gin_net.cuh $(BUILD_DIR)/fifo.o
	$(NVCC) $(DEVFLAGS) -I$(ROOT) -I$(NCCL_INCLUDE_DIR) -DNCCL_CHECK_CUDACC=1 -DUCCL_GIN_WITH_NCCL_GIN=1 -MMD -MP $< $(BUILD_DIR)/fifo.o $(LIBS) -o $@

adapter-compile-fail: | $(BUILD_DIR)
	@if $(NVCC) $(DEVFLAGS) -I$(ROOT) -I$(NCCL_INCLUDE_DIR) -DNCCL_CHECK_CUDACC=1 -DUCCL_GIN_WITH_NCCL_GIN=1 -c unsupported_signal_coop.cu -o $(BUILD_DIR)/unsupported_signal_coop.o > $(BUILD_DIR)/unsupported_signal_coop.log 2>&1; then echo "FAIL: CTA signal compiled"; exit 1; fi
	@grep -q 'signal cooperation supports Thread/Warp only' $(BUILD_DIR)/unsupported_signal_coop.log || (cat $(BUILD_DIR)/unsupported_signal_coop.log; exit 1)

$(BUILD_DIR)/network_gate.o: network_gate_compile.cu ../coop_flush_gate.cuh | $(BUILD_DIR)
	$(NVCC) $(DEVFLAGS) -I$(ROOT) -I$(NCCL_INCLUDE_DIR) -DNCCL_CHECK_CUDACC=1 -DUCCL_GIN_WITH_NCCL_GIN=1 -MMD -MP -c $< -o $@

$(BUILD_DIR)/coop_flush: coop_flush.cu queue_fixture.hpp $(BUILD_DIR)/fifo.o
	@test -f "$(NCCL_INCLUDE_DIR)/nccl_device.h" || (echo "Set NCCL_INCLUDE_DIR to the NCCL device headers"; exit 1)
	$(NVCC) $(DEVFLAGS) -I$(ROOT) -I$(NCCL_INCLUDE_DIR) -DNCCL_CHECK_CUDACC=1 -DUCCL_GIN_WITH_NCCL_GIN=1 -MMD -MP $< $(BUILD_DIR)/fifo.o $(LIBS) -o $@

compile-fail: | $(BUILD_DIR)
	@if $(NVCC) $(DEVFLAGS) -I$(ROOT) -I$(NCCL_INCLUDE_DIR) -DNCCL_CHECK_CUDACC=1 -DUCCL_GIN_WITH_NCCL_GIN=1 -c unsupported_coop.cu -o $(BUILD_DIR)/unsupported_coop.o > $(BUILD_DIR)/unsupported_coop.log 2>&1; then echo "FAIL: CTA flush compiled"; exit 1; fi
	@grep -q 'cooperative flush supports ncclCoopThread' $(BUILD_DIR)/unsupported_coop.log || (cat $(BUILD_DIR)/unsupported_coop.log; exit 1)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/fifo.o: $(GIN)/transport/fifo.cpp | $(BUILD_DIR)
	$(CXX) $(HOSTFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%: %.cu queue_fixture.hpp $(BUILD_DIR)/fifo.o
	$(NVCC) $(DEVFLAGS) -DUCCL_GIN_WITH_NCCL_GIN=0 -MMD -MP $< $(BUILD_DIR)/fifo.o $(LIBS) -o $@

clean:
	rm -rf build

-include $(BUILD_DIR)/*.d
