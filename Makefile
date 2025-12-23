#---------------------------------------------------------------------
# Makefile for vanitysearch
#
# Author : Jean-Luc PONS
# Optimizations by: FixedPaul
# Additional improvements for modern GPU architectures
#
# Build options:
#   make              - Build with default optimizations
#   make debug=1      - Build with debug symbols
#   make ARCH=sm_89   - Build for specific GPU architecture
#   make clean        - Clean build artifacts
#---------------------------------------------------------------------

SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp Wildcard.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        GPU/GPUEngine.o Bech32.o Wildcard.o)

#---------------------------------------------------------------------
# Compiler settings
#---------------------------------------------------------------------
CXX        = g++-9
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++-9
NVCC       = $(CUDA)/bin/nvcc

#---------------------------------------------------------------------
# Optimization flags
#---------------------------------------------------------------------
# CPU optimization flags for modern processors
CPU_OPT_FLAGS = -march=native -mtune=native -ffast-math -funroll-loops

# SIMD optimization flags
SIMD_FLAGS = -mssse3 -msse4.1 -msse4.2

# OpenMP for parallel CPU operations
OMP_FLAGS = -fopenmp

#---------------------------------------------------------------------
# Build configuration
#---------------------------------------------------------------------
ifdef debug
CXXFLAGS   = $(SIMD_FLAGS) -Wno-write-strings -g -I. -I$(CUDA)/include $(OMP_FLAGS) -DDEBUG
NVCC_FLAGS = -G -lineinfo
else
CXXFLAGS   = $(SIMD_FLAGS) -Wno-write-strings -O3 $(CPU_OPT_FLAGS) -I. -I$(CUDA)/include $(OMP_FLAGS) -DNDEBUG
NVCC_FLAGS = -O3 --use_fast_math
endif

LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart $(OMP_FLAGS)

#---------------------------------------------------------------------
# GPU Architecture Support
#
# Supported architectures:
#   sm_60 - Pascal (GTX 1060, 1070, 1080, P100)
#   sm_61 - Pascal (GTX 1050, 1080 Ti)
#   sm_70 - Volta (V100, Titan V)
#   sm_75 - Turing (RTX 2060, 2070, 2080, T4)
#   sm_80 - Ampere (A100, A30)
#   sm_86 - Ampere (RTX 3060, 3070, 3080, 3090)
#   sm_89 - Ada Lovelace (RTX 4060, 4070, 4080, 4090)
#   sm_90 - Hopper (H100)
#   sm_100 - Blackwell (RTX 5090, B100) [Experimental]
#---------------------------------------------------------------------

# Default: build for all common architectures
CUDA_ARCH = -gencode=arch=compute_60,code=sm_60 \
            -gencode=arch=compute_61,code=sm_61 \
            -gencode=arch=compute_70,code=sm_70 \
            -gencode=arch=compute_75,code=sm_75 \
            -gencode=arch=compute_80,code=sm_80 \
            -gencode=arch=compute_86,code=sm_86 \
            -gencode=arch=compute_89,code=sm_89

# Uncomment for Hopper support (requires CUDA 12+)
# CUDA_ARCH += -gencode=arch=compute_90,code=sm_90

# Uncomment for Blackwell support (requires CUDA 12.4+)
# CUDA_ARCH += -gencode=arch=compute_100,code=sm_100

# Allow single architecture override
ifdef ARCH
CUDA_ARCH = -gencode=arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH)
endif

# PTX for forward compatibility
CUDA_ARCH += -gencode=arch=compute_89,code=compute_89

#---------------------------------------------------------------------
# NVCC optimization settings
#---------------------------------------------------------------------
# maxrregcount=0 allows NVCC to use optimal register count
# --ptxas-options=-v shows register usage
NVCC_OPT = -maxrregcount=0 --ptxas-options=-v

# Compiler compatibility
NVCC_COMPAT = --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64

#---------------------------------------------------------------------
# Build rules
#---------------------------------------------------------------------

ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT) $(NVCC_COMPAT) -g \
		-I$(CUDA)/include $(CUDA_ARCH) \
		-o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT) $(NVCC_COMPAT) \
		-I$(CUDA)/include $(CUDA_ARCH) \
		-o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

#---------------------------------------------------------------------
# Targets
#---------------------------------------------------------------------

all: VanitySearch

VanitySearch: $(OBJET)
	@echo "============================================"
	@echo "Building VanitySearch..."
	@echo "============================================"
	$(CXX) $(OBJET) $(LFLAGS) -o vanitysearch
	@echo "============================================"
	@echo "Build complete: ./vanitysearch"
	@echo "============================================"

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p hash

#---------------------------------------------------------------------
# Utility targets
#---------------------------------------------------------------------

clean:
	@echo "Cleaning build artifacts..."
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o
	@rm -f vanitysearch
	@echo "Clean complete."

# Display GPU info (nvidia-smi)
gpu-info:
	@$(CUDA)/bin/nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Show CUDA version
cuda-version:
	@$(NVCC) --version

# Build GPU diagnostic utility
gpu-diag: GPU/gpu_info.cu
	@echo "Building GPU diagnostic utility..."
	$(NVCC) -O3 -o gpu_diag GPU/gpu_info.cu
	@echo "Run './gpu_diag' to see detailed GPU information"

# Help target
help:
	@echo "VanitySearch Build System"
	@echo "========================="
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build with default optimizations"
	@echo "  make debug=1      - Build with debug symbols"
	@echo "  make ARCH=sm_89   - Build for specific GPU architecture"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make gpu-info     - Show GPU information (nvidia-smi)"
	@echo "  make gpu-diag     - Build GPU diagnostic utility"
	@echo "  make cuda-version - Show CUDA version"
	@echo ""
	@echo "Supported GPU Architectures:"
	@echo "  sm_60  - Pascal (GTX 1060/1070/1080)"
	@echo "  sm_61  - Pascal (GTX 1050/1080 Ti)"
	@echo "  sm_70  - Volta (V100)"
	@echo "  sm_75  - Turing (RTX 20xx)"
	@echo "  sm_80  - Ampere (A100)"
	@echo "  sm_86  - Ampere (RTX 30xx)"
	@echo "  sm_89  - Ada Lovelace (RTX 40xx)"
	@echo "  sm_90  - Hopper (H100)"
	@echo "  sm_100 - Blackwell (RTX 50xx)"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Run './vanitysearch -l' to see your GPU"
	@echo "  2. Build for your GPU: make ARCH=sm_XX"
	@echo "  3. Run: ./vanitysearch -gpuId 0 -start HEX -range N ADDRESS"
	@echo ""

.PHONY: all clean gpu-info gpu-diag cuda-version help
