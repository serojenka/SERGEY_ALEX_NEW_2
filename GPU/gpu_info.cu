/*
 * GPU Information and Diagnostic Utility
 * Part of VanitySearch-Bitcrack
 *
 * This utility provides detailed GPU information to help users
 * optimize their build and runtime configuration.
 *
 * Compile: nvcc -o gpu_info gpu_info.cu
 * Run: ./gpu_info
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// GPU Architecture to CUDA Cores mapping
int ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        // Pascal
        {0x60,  64},   // GP100 (Tesla P100)
        {0x61, 128},   // GP104 (GTX 1080)
        {0x62, 128},   // GP10B (Tegra)
        // Volta
        {0x70,  64},   // GV100 (Tesla V100)
        {0x72,  64},   // GV10B (Jetson AGX Xavier)
        // Turing
        {0x75,  64},   // TU102/TU104/TU106 (RTX 20xx)
        // Ampere
        {0x80,  64},   // GA100 (A100)
        {0x86, 128},   // GA102 (RTX 30xx)
        {0x87, 128},   // GA10B (Jetson Orin)
        // Ada Lovelace
        {0x89, 128},   // AD102 (RTX 40xx)
        // Hopper
        {0x90, 128},   // GH100 (H100)
        // Blackwell
        {0xa0, 128},   // GB100 (B100, RTX 50xx)
        {0xa1, 128},
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // Default for unknown architectures
    if (major >= 9) return 128;
    if (major >= 8) return 128;
    if (major >= 7) return 64;
    return 64;
}

const char* GetArchitectureName(int major, int minor) {
    if (major >= 10) return "Blackwell";
    if (major == 9) return "Hopper";
    if (major == 8 && minor >= 9) return "Ada Lovelace";
    if (major == 8) return "Ampere";
    if (major == 7 && minor >= 5) return "Turing";
    if (major == 7) return "Volta";
    if (major == 6) return "Pascal";
    if (major == 5) return "Maxwell";
    if (major == 3) return "Kepler";
    return "Unknown";
}

void printDeviceInfo(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    int coresPerSM = ConvertSMVer2Cores(prop.major, prop.minor);
    int totalCores = coresPerSM * prop.multiProcessorCount;
    const char* archName = GetArchitectureName(prop.major, prop.minor);

    printf("GPU #%d: %s\n", deviceId, prop.name);
    printf("============================================\n");
    printf("Architecture:         %s\n", archName);
    printf("Compute Capability:   %d.%d (sm_%d%d)\n",
           prop.major, prop.minor, prop.major, prop.minor);
    printf("\n");

    printf("-- Compute Resources --\n");
    printf("Multiprocessors:      %d\n", prop.multiProcessorCount);
    printf("CUDA Cores/SM:        %d\n", coresPerSM);
    printf("Total CUDA Cores:     %d\n", totalCores);
    printf("GPU Clock Rate:       %.2f GHz\n", prop.clockRate / 1e6);
    printf("Max Threads/Block:    %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads/SM:       %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Warp Size:            %d\n", prop.warpSize);
    printf("\n");

    printf("-- Memory --\n");
    printf("Total Global Memory:  %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memory Clock Rate:    %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width:     %d-bit\n", prop.memoryBusWidth);
    printf("Peak Memory BW:       %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("L2 Cache Size:        %d KB\n", prop.l2CacheSize / 1024);
    printf("Shared Memory/Block:  %zu bytes\n", prop.sharedMemPerBlock);
    printf("Shared Memory/SM:     %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Registers/Block:      %d\n", prop.regsPerBlock);
    printf("Registers/SM:         %d\n", prop.regsPerMultiprocessor);
    printf("\n");

    printf("-- Features --\n");
    printf("Concurrent Kernels:   %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async Engine Count:   %d\n", prop.asyncEngineCount);
    printf("ECC Enabled:          %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("Unified Addressing:   %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("Managed Memory:       %s\n", prop.managedMemory ? "Yes" : "No");
    printf("\n");

    // Performance estimate for VanitySearch workload
    double estimatedMKeys = (double)totalCores * (prop.clockRate / 1e6) * 0.0025;
    printf("-- VanitySearch Estimate --\n");
    printf("Estimated Performance: ~%.0f MKey/s\n", estimatedMKeys);
    printf("Recommended Build:     make ARCH=sm_%d%d\n", prop.major, prop.minor);
    printf("\n");
}

void printBuildRecommendations(int numDevices) {
    printf("============================================\n");
    printf("Build Recommendations:\n");
    printf("============================================\n\n");

    if (numDevices == 0) {
        printf("No CUDA GPUs detected. Please ensure:\n");
        printf("  1. NVIDIA GPU is installed\n");
        printf("  2. NVIDIA drivers are installed\n");
        printf("  3. CUDA toolkit is installed\n");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Use first GPU for recommendations

    printf("For your GPU(s), use:\n\n");
    printf("  # Build for your specific GPU (best performance):\n");
    printf("  make clean && make ARCH=sm_%d%d\n\n", prop.major, prop.minor);

    printf("  # Build for all common architectures:\n");
    printf("  make clean && make\n\n");

    printf("  # Debug build with performance timing:\n");
    printf("  make clean && make debug=1\n\n");

    printf("  # Check your GPU info:\n");
    printf("  ./vanitysearch -l\n\n");

    printf("============================================\n");
    printf("Runtime Tips:\n");
    printf("============================================\n");
    printf("  - Use '-gpuId N' to select specific GPU\n");
    printf("  - Increase '-m' for many target addresses\n");
    printf("  - Use '-random' for large key ranges\n");
    printf("  - Use '-backup' for resumable searches\n");
}

int main(int argc, char** argv) {
    int deviceCount = 0;
    int driverVersion = 0, runtimeVersion = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        printf("\nPlease ensure NVIDIA drivers and CUDA are properly installed.\n");
        return 1;
    }

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("\n");
    printf("============================================\n");
    printf("VanitySearch-Bitcrack GPU Diagnostic Tool\n");
    printf("============================================\n\n");

    printf("CUDA Driver Version:   %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version:  %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("Number of GPUs:        %d\n", deviceCount);
    printf("\n");

    if (deviceCount == 0) {
        printf("No CUDA-capable GPU detected!\n\n");
    } else {
        for (int i = 0; i < deviceCount; i++) {
            printDeviceInfo(i);
        }
    }

    printBuildRecommendations(deviceCount);

    return 0;
}
