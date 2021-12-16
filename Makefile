BUILD_DIR := ./build
BIN_DIR := $(BUILD_DIR)/bin
SRC_DIR := ./src

NVCC := nvcc
NVCCFLAGS := -gencode arch=compute_50,code=sm_50

.PHONY: all clean

all: $(BIN_DIR) \
	 $(BIN_DIR)/miracle

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/miracle: $(BIN_DIR) $(SRC_DIR)/driver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(SRC_DIR)/miracle.cuh $(BUILD_DIR)/miracle.o $(SRC_DIR)/miracle_dynamic.cuh $(BUILD_DIR)/miracle_dynamic.o $(SRC_DIR)/miracle_gpu.cuh $(BUILD_DIR)/miracle_gpu.o $(SRC_DIR)/launch_parameters_gpu.cuh $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/driver.cu $(BUILD_DIR)/cnf_formula.o $(BUILD_DIR)/cnf_formula_gpu.o $(BUILD_DIR)/miracle.o $(BUILD_DIR)/miracle_dynamic.o $(BUILD_DIR)/miracle_gpu.o $(BUILD_DIR)/utils.o -o $@

$(BUILD_DIR)/cnf_formula.o: $(BIN_DIR) $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/cnf_formula.cu $(SRC_DIR)/cnf_formula_types.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/cnf_formula.cu -o $@

$(BUILD_DIR)/cnf_formula_gpu.o: $(BIN_DIR) $(SRC_DIR)/cnf_formula_gpu.cuh $(SRC_DIR)/cnf_formula_gpu.cu $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/cnf_formula_gpu.cu -o $@

$(BUILD_DIR)/miracle.o: $(BIN_DIR) $(SRC_DIR)/miracle.cuh $(SRC_DIR)/miracle.cu $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/miracle.cu -o $@

$(BUILD_DIR)/miracle_dynamic.o: $(BIN_DIR) $(SRC_DIR)/miracle_dynamic.cuh $(SRC_DIR)/miracle_dynamic.cu $(SRC_DIR)/cnf_formula.cuh $(SRC_DIR)/utils.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/miracle_dynamic.cu -o $@

$(BUILD_DIR)/miracle_gpu.o: $(BIN_DIR) $(SRC_DIR)/miracle_gpu.cuh $(SRC_DIR)/miracle_gpu.cu $(SRC_DIR)/cnf_formula_gpu.cuh $(SRC_DIR)/miracle.cuh $(SRC_DIR)/utils.cuh $(SRC_DIR)/launch_parameters_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/miracle_gpu.cu -o $@

$(BUILD_DIR)/utils.o: $(BIN_DIR) $(SRC_DIR)/utils.cuh $(SRC_DIR)/utils.cu $(SRC_DIR)/launch_parameters_gpu.cuh
	$(NVCC) $(NVCCFLAGS) -c $(SRC_DIR)/utils.cu -o $@


clean:
	rm -rf $(BUILD_DIR)
