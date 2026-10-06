NVCC ?= nvcc
SM ?= 120
ROOT := ../../..
BUILD := build/sm$(SM)
FLAGS := -O3 -std=c++17 -arch=sm_$(SM) -I$(ROOT)/thirdparty/dietgpu -I$(ROOT)/include
CODEC := $(ROOT)/thirdparty/dietgpu/dietgpu/float/libdietgpu_float.so

.PHONY: all codec
all: $(BUILD)/probe $(BUILD)/codec_test $(BUILD)/staged_test $(BUILD)/libstaged_copy.so

$(BUILD):
	mkdir -p $@

$(BUILD)/probe: probe.cu ../../local/cuda_utils.h | $(BUILD)
	$(NVCC) $(FLAGS) $< -o $@

codec:
	$(MAKE) -C $(ROOT)/thirdparty/dietgpu/dietgpu/float -f Makefile.cuda GPU_ARCH=sm_$(SM) NVCC=$(NVCC) -j2

$(BUILD)/codec_test: codec_test.cu codec ../../local/cuda_utils.h | $(BUILD)
	$(NVCC) $(FLAGS) $< $(CODEC) -Xlinker -rpath -Xlinker $(abspath $(dir $(CODEC))) -o $@

$(BUILD)/staged_copy.o: ../../local/staged_copy.cc ../../local/staged_copy.h ../../local/cuda_utils.h | $(BUILD)
	$(NVCC) $(FLAGS) -x cu -Xcompiler -pthread,-fPIC -c $< -o $@

$(BUILD)/staged_test: staged_test.cu $(BUILD)/staged_copy.o codec | $(BUILD)
	$(NVCC) $(FLAGS) $< $(BUILD)/staged_copy.o $(CODEC) -Xcompiler -pthread -Xlinker -rpath -Xlinker $(abspath $(dir $(CODEC))) -o $@

$(BUILD)/libstaged_copy.so: ../../local/binding.cu $(BUILD)/staged_copy.o codec | $(BUILD)
	$(NVCC) $(FLAGS) -shared -Xcompiler -pthread,-fPIC $< $(BUILD)/staged_copy.o $(CODEC) -Xlinker -rpath -Xlinker $(abspath $(dir $(CODEC))) -o $@
