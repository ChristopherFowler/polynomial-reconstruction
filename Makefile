# Makefile for standalone C++ profile program with CUDA

# Compiler
NVCC = nvcc
CXX = g++

# Flags
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++11
CXX_FLAGS = -O3 -std=c++11
CUDA_LIBS = -lcudart -lcusolver

# Directories
BUILD_DIR = build_standalone

# Targets
TARGET = profile_poly3d_cpp
MINIMAL_TARGET = minimal_example
OPTIMIZED_TARGET = optimized_example
COLORMODEL_TARGET = colormodel_integration_example

# Source files
CUDA_SRC = poly_cuda.cu
CPP_SRC = profile_poly3d.cpp
MINIMAL_SRC = minimal_example.cpp
OPTIMIZED_SRC = optimized_example.cpp
COLORMODEL_SRC = colormodel_integration_example.cpp
CUDA_OBJ = $(BUILD_DIR)/poly_cuda.o
CPP_OBJ = $(BUILD_DIR)/profile_poly3d.o
MINIMAL_OBJ = $(BUILD_DIR)/minimal_example.o
OPTIMIZED_OBJ = $(BUILD_DIR)/optimized_example.o
COLORMODEL_OBJ = $(BUILD_DIR)/colormodel_integration_example.o

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA code
$(CUDA_OBJ): $(CUDA_SRC) poly_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_SRC) -o $(CUDA_OBJ)

# Compile C++ code
$(CPP_OBJ): $(CPP_SRC) poly_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $(CPP_SRC) -o $(CPP_OBJ)

# Link everything together
$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_OBJ) $(CPP_OBJ) -o $(TARGET) $(CUDA_LIBS)

# Compile minimal example
$(MINIMAL_OBJ): $(MINIMAL_SRC) poly_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $(MINIMAL_SRC) -o $(MINIMAL_OBJ)

# Link minimal example
$(MINIMAL_TARGET): $(CUDA_OBJ) $(MINIMAL_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_OBJ) $(MINIMAL_OBJ) -o $(MINIMAL_TARGET) $(CUDA_LIBS)

# Compile optimized example
$(OPTIMIZED_OBJ): $(OPTIMIZED_SRC) poly_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $(OPTIMIZED_SRC) -o $(OPTIMIZED_OBJ)

# Link optimized example
$(OPTIMIZED_TARGET): $(CUDA_OBJ) $(OPTIMIZED_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_OBJ) $(OPTIMIZED_OBJ) -o $(OPTIMIZED_TARGET) $(CUDA_LIBS)

# Compile ColorModel integration example
$(COLORMODEL_OBJ): $(COLORMODEL_SRC) poly_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $(COLORMODEL_SRC) -o $(COLORMODEL_OBJ)

# Link ColorModel integration example
$(COLORMODEL_TARGET): $(CUDA_OBJ) $(COLORMODEL_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_OBJ) $(COLORMODEL_OBJ) -o $(COLORMODEL_TARGET) $(CUDA_LIBS)

# Build both targets
both: $(TARGET) $(MINIMAL_TARGET)

# Build all targets
all: $(TARGET) $(MINIMAL_TARGET) $(OPTIMIZED_TARGET) $(COLORMODEL_TARGET)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(MINIMAL_TARGET) $(OPTIMIZED_TARGET) $(COLORMODEL_TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run minimal example
minimal: $(MINIMAL_TARGET)
	./$(MINIMAL_TARGET)

# Run optimized example  
optimized: $(OPTIMIZED_TARGET)
	./$(OPTIMIZED_TARGET)

# Run ColorModel integration example
colormodel: $(COLORMODEL_TARGET)
	./$(COLORMODEL_TARGET)

.PHONY: all clean run minimal optimized both colormodel
