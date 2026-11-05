# Optional module: the existing single-node/fusion targets do not link this library.
ROUTER_BUILD := $(BUILD)/cpu_router
ROUTER_INC := -Iinclude -Iinclude/uccl_lite/cpu_router -I$(CUDA_HOME)/include
ROUTER_HOST := cpu_executor cpu_router_c memory router_core signal backend_policy collective_dispatch
ROUTER_CUDA := channel packet_pipeline
ROUTER_OBJECTS := $(addprefix $(ROUTER_BUILD)/,$(addsuffix .o,$(ROUTER_HOST) $(ROUTER_CUDA)))
ROUTER_STATIC := $(ROUTER_BUILD)/libuccl_lite_cpu_router.a
ROUTER_SHARED := $(ROUTER_BUILD)/libuccl_lite_cpu_router.so
ROUTER_LIB := $(ROUTER_STATIC) -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl -Wl,-rpath,$(CUDA_HOME)/lib64
ROUTER_CUDA_TESTS := primitives_check primitive_example router_check router_graph_check
ROUTER_HOST_TESTS := cpu_bf16_check router_core_check dispatcher_check

.PHONY: cpu-router adaptive-collectives router-checks router-policy-check
cpu-router: $(ROUTER_STATIC) $(ROUTER_SHARED)
adaptive-collectives: $(BUILD)/uccl_lite_collectives
router-policy-check: $(BUILD)/uccl_lite_router_policy_check
router-checks: cpu-router router-policy-check $(addprefix $(BUILD)/uccl_lite_,$(ROUTER_CUDA_TESTS) $(ROUTER_HOST_TESTS))

$(ROUTER_BUILD):
	mkdir -p $@

$(addprefix $(ROUTER_BUILD)/,$(addsuffix .o,$(ROUTER_HOST))): $(ROUTER_BUILD)/%.o: src/cpu_router/%.cc | $(ROUTER_BUILD)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP $(ROUTER_INC) -c $< -o $@

$(addprefix $(ROUTER_BUILD)/,$(addsuffix .o,$(ROUTER_CUDA))): $(ROUTER_BUILD)/%.o: src/cpu_router/%.cu | $(ROUTER_BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -Xcompiler=-fPIC $(FUSION_FLAGS) $(ROUTER_INC) -I$(CUTLASS_HOME)/include -c $< -o $@

$(ROUTER_STATIC): $(ROUTER_OBJECTS)
	$(AR) rcs $@ $^

$(ROUTER_SHARED): $(ROUTER_OBJECTS)
	$(CXX) -shared $^ -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(addprefix $(BUILD)/uccl_lite_,$(ROUTER_HOST_TESTS)): $(BUILD)/uccl_lite_%: tests/cpu_router/%.cc $(ROUTER_STATIC) | $(BUILD)
	$(CXX) -O3 -std=c++17 -MMD -MP $(ROUTER_INC) $< $(ROUTER_LIB) -o $@

$(addprefix $(BUILD)/uccl_lite_,$(ROUTER_CUDA_TESTS)): $(BUILD)/uccl_lite_%: tests/cpu_router/%.cu $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) $< $(ROUTER_STATIC) -Xcompiler=-pthread -ldl -o $@

$(BUILD)/uccl_lite_router_policy_check: tests/cpu_router/backend_policy_check.cc src/cpu_router/backend_policy.cc include/uccl_lite/backend_policy.hpp | $(BUILD)
	$(CXX) -O2 -std=c++17 -Iinclude tests/cpu_router/backend_policy_check.cc src/cpu_router/backend_policy.cc -o $@

$(BUILD)/uccl_lite_collectives: benchmarks/cpu_router/collectives_bench.cu $(ROUTER_STATIC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(ROUTER_INC) -I$(NCCL_HOME)/include $< $(ROUTER_STATIC) -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib -Xcompiler=-pthread -ldl -o $@

-include $(wildcard $(ROUTER_BUILD)/*.d)
