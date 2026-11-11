# UCCL Lite standalone benchmarks. External dependencies are never vendored.
.DEFAULT_GOAL := single-node

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_89
MSCCLPP_HOME ?= $(HOME)/third_party/mscclpp
MSCCLPP_BUILD ?= $(MSCCLPP_HOME)/build
MSCCLPP_IB_BUILD ?= $(MSCCLPP_HOME)/build-ib
NCCL_HOME ?= $(HOME)/third_party/nccl/build
CUTLASS_HOME ?= $(HOME)/third_party/cutlass
MPICXX ?= mpicxx
BUILD := build
EXTRA_NVCC_FLAGS ?=

NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -MMD -MP $(EXTRA_NVCC_FLAGS)
MSCCLPP_INC = -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_BUILD)/include
MSCCLPP_LIB = -L$(MSCCLPP_BUILD)/lib -lmscclpp -Xlinker -rpath=$(MSCCLPP_BUILD)/lib
IB_INC = -I$(MSCCLPP_HOME)/include -I$(MSCCLPP_IB_BUILD)/include
IB_LIB = -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Xlinker -rpath=$(MSCCLPP_IB_BUILD)/lib
MPI_INC = $(shell $(MPICXX) --showme:compile)
# Open MPI emits host-compiler -Wl options; nvcc needs -Xlinker instead.
MPI_LIB = $(shell $(MPICXX) --showme:link | sed 's/-Wl,/-Xlinker /g')
FUSION_INC = $(MSCCLPP_INC) -Iinclude/uccl_lite -I$(CUTLASS_HOME)/include -I$(NCCL_HOME)/include
FUSION_FLAGS := --expt-relaxed-constexpr -diag-suppress 20012,20014
FUSION_LIB = $(MSCCLPP_LIB) -L$(NCCL_HOME)/lib -lnccl -lcublas -Xlinker -rpath=$(NCCL_HOME)/lib

COLLECTIVES := allreduce alltoall
PEER_COLLECTIVES := peer_ag_rs
NCCL_CONTROLS := nccl_4l4_mpi_baseline nccl_2x4l4_baseline
MULTINODE := multinode_allreduce multinode_cpu_router host_hierarchical_allreduce
FUSION := dense_fusion fused_moe fused_moe_layer fused_moe_layer_qwen router_fusion_8l4
PROBES := gemm_probe mma_gemm_probe mma_peak_probe moe_overlap_probe cutlass_gemm_probe hardware_profiler
ALL_NAMES := $(COLLECTIVES) $(PEER_COLLECTIVES) $(NCCL_CONTROLS) $(MULTINODE) host_allreduce $(FUSION) $(PROBES)
BIN = $(addprefix $(BUILD)/uccl_lite_,$(1))

.PHONY: all single-node peer-collectives eight-gpu-alltoall duplex-allgather-8l4 fusion probes multinode headers help $(ALL_NAMES)
all: single-node peer-collectives fusion probes headers
single-node: $(call BIN,$(COLLECTIVES))
peer-collectives: $(call BIN,$(PEER_COLLECTIVES))
eight-gpu-alltoall: $(BUILD)/uccl_lite_alltoall_8l4
duplex-allgather-8l4: $(BUILD)/uccl_lite_duplex_allgather_8l4
fusion: $(call BIN,$(FUSION))
probes: $(call BIN,$(PROBES))
multinode: $(call BIN,$(MULTINODE) host_allreduce)
headers: $(BUILD)/header_check.o
$(ALL_NAMES): %: $(BUILD)/uccl_lite_%

$(BUILD):
	mkdir -p $@

$(call BIN,allreduce alltoall): $(BUILD)/uccl_lite_%: src/collectives/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(MSCCLPP_INC) $< -o $@ $(MSCCLPP_LIB) -lcudart -lpthread

$(BUILD)/uccl_lite_peer_ag_rs: src/collectives/peer_ag_rs.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(MSCCLPP_INC) -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib

$(BUILD)/uccl_lite_nccl_4l4_mpi_baseline: benchmarks/collectives/nccl_4l4_mpi_baseline.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/uccl_lite_nccl_2x4l4_baseline: benchmarks/collectives/nccl_2x4l4_baseline.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

$(BUILD)/uccl_lite_alltoall_8l4: benchmarks/collectives/alltoall_8l4_bench.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread

$(BUILD)/uccl_lite_duplex_allgather_8l4: benchmarks/cpu_router/duplex_allgather_8l4.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

$(call BIN,$(MULTINODE)): $(BUILD)/uccl_lite_%: src/collectives/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(IB_INC) $(MPI_INC) $< -o $@ $(IB_LIB) $(MPI_LIB) -lcudart -lpthread

$(BUILD)/uccl_lite_multinode_cpu_router: src/collectives/multinode_allreduce.cu

# This fallback deliberately has no MSCCL++ dependency or GPU-memory registration.
$(BUILD)/uccl_lite_host_allreduce: src/collectives/host_allreduce.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(MPI_INC) $< -o $@ $(MPI_LIB) -lcudart -lpthread

$(call BIN,dense_fusion fused_moe fused_moe_layer): $(BUILD)/uccl_lite_%: benchmarks/fusion/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(FUSION_INC) $< -o $@ $(FUSION_LIB) -lcudart -lpthread

$(BUILD)/uccl_lite_router_fusion_8l4: benchmarks/fusion/router_fusion_8l4.cu $(BUILD)/cpu_router/libuccl_lite_cpu_router.a | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -o $@ -L$(NCCL_HOME)/lib -lnccl -lcublas -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl

$(BUILD)/uccl_lite_fused_moe_layer_qwen: benchmarks/fusion/fused_moe_layer.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DMOE_QWEN36 $(FUSION_FLAGS) $(FUSION_INC) $< -o $@ $(FUSION_LIB) -lcudart -lpthread

$(call BIN,gemm_probe mma_gemm_probe mma_peak_probe moe_overlap_probe): $(BUILD)/uccl_lite_%: benchmarks/probes/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $< -o $@ -lcublas

$(BUILD)/uccl_lite_cutlass_gemm_probe: benchmarks/probes/cutlass_gemm_probe.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) -Iinclude/uccl_lite -I$(CUTLASS_HOME)/include $< -o $@ -lcublas

$(BUILD)/uccl_lite_hardware_profiler: benchmarks/probes/hardware_profiler.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $< -o $@ -Xcompiler=-pthread

$(BUILD)/header_check.o: benchmarks/header_check.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(FUSION_FLAGS) $(FUSION_INC) -c $< -o $@

help:
	@printf '%s\n' 'UCCL Lite targets:' '  make single-node  (default): AllReduce and All-to-All' '  make peer-collectives: four-GPU AllGather/ReduceScatter + NCCL comparison' '  make fusion: dense and expert-parallel MoE benchmarks' '  make probes: hardware topology/bandwidth and GEMM/overlap diagnostics' '  make headers: compile-check reusable MoE headers' '  make all: all of the above' '  make multinode: two-node benchmarks (Open MPI required)' '  make <name>: build one executable, e.g. multinode_allreduce' 'See docs/build.md for dependency paths and launch commands.'
	@printf '%s\n' '  make cpu-router: optional static/shared CPU-router module' '  make local-collectives: two-NUMA BF16 fallback library' '  make adaptive-collectives: BF16 dispatcher + local/router calibration (optional NCCL control)' '  make router-checks local-collectives-check: core, policy, eager, and graph correctness executables'
	@printf '%s\n' '  make eight-gpu-alltoall: BF16 host/peer paths against NCCL on two four-GPU NUMA groups'
	@printf '%s\n' '  make nccl_4l4_mpi_baseline nccl_2x4l4_baseline: checked FP32 AllReduce and AR/AG/RS controls'

include src/cpu_router/module.mk

-include $(wildcard $(BUILD)/*.d)
