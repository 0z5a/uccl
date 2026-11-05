# Standalone paper-reference transport; no CPU-router, NCCL or MSCCL++ link.
HOST_STAGED_BUILD := $(BUILD)/host_staged
HOST_STAGED_LIB := $(HOST_STAGED_BUILD)/libuccl_lite_host_staged.a

.PHONY: host-staged-reference host-staged-checks
host-staged-reference: $(HOST_STAGED_LIB)
host-staged-checks: $(BUILD)/uccl_lite_host_staged_check $(BUILD)/uccl_lite_host_staged_moe_check

.PHONY: host-staged-benchmark
host-staged-benchmark: $(BUILD)/uccl_lite_host_staged_compare

$(BUILD)/uccl_lite_host_staged_compare: benchmarks/host_staged/compare.cu $(HOST_STAGED_LIB) $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -I$(NCCL_HOME)/include $< $(HOST_STAGED_LIB) $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@

$(HOST_STAGED_BUILD):
	mkdir -p $@

$(HOST_STAGED_BUILD)/ep_transport.o: src/host_staged/ep_transport.cu include/uccl_lite/host_staged/ep_transport.hpp | $(HOST_STAGED_BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude -Xcompiler=-fPIC -c $< -o $@

$(HOST_STAGED_LIB): $(HOST_STAGED_BUILD)/ep_transport.o
	$(AR) rcs $@ $^

$(BUILD)/uccl_lite_host_staged_check: tests/host_staged/transport_check.cu $(HOST_STAGED_LIB) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< $(HOST_STAGED_LIB) -o $@

$(BUILD)/uccl_lite_host_staged_moe_check: tests/host_staged/moe_check.cu $(HOST_STAGED_LIB) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Iinclude $< $(HOST_STAGED_LIB) -o $@

-include $(wildcard $(HOST_STAGED_BUILD)/*.d)
