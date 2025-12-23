# GPU Optimization Guide

**CUDA Performance Optimization for Elliptic Curve Cryptography**

## Table of Contents

1. [GPU Architecture Overview](#1-gpu-architecture-overview)
2. [Memory Hierarchy Optimization](#2-memory-hierarchy-optimization)
3. [Kernel Optimization Techniques](#3-kernel-optimization-techniques)
4. [256-bit Arithmetic on GPU](#4-256-bit-arithmetic-on-gpu)
5. [Performance Profiling](#5-performance-profiling)
6. [Architecture-Specific Tuning](#6-architecture-specific-tuning)

---

## 1. GPU Architecture Overview

### 1.1 NVIDIA GPU Hierarchy

```
GPU
├── Streaming Multiprocessors (SMs)
│   ├── CUDA Cores
│   ├── Special Function Units (SFUs)
│   ├── Load/Store Units
│   ├── Warp Schedulers
│   └── Register File
├── L2 Cache
├── Memory Controllers
└── Global Memory (VRAM)
```

### 1.2 Execution Model

**Grid → Blocks → Warps → Threads**

```
Grid Configuration (VanitySearch):
- Blocks: numSMs × 128
- Threads per block: 256
- Warps per block: 8

Thread indexing:
  globalIdx = blockIdx.x * blockDim.x + threadIdx.x
```

### 1.3 Architecture Comparison

| Architecture | SMs | Cores/SM | L2 Cache | Memory BW |
|-------------|-----|----------|----------|-----------|
| Pascal (sm_60) | 56 | 64 | 4 MB | 720 GB/s |
| Turing (sm_75) | 68 | 64 | 6 MB | 616 GB/s |
| Ampere (sm_86) | 84 | 128 | 6 MB | 936 GB/s |
| Ada (sm_89) | 128 | 128 | 72 MB | 1008 GB/s |

---

## 2. Memory Hierarchy Optimization

### 2.1 Memory Types and Usage

| Memory Type | Size | Latency | Bandwidth | Use Case |
|-------------|------|---------|-----------|----------|
| Registers | ~256KB/SM | 1 cycle | Highest | Local variables |
| Shared | 48-164KB/SM | ~20 cycles | 1.5 TB/s | Thread cooperation |
| L1 Cache | 32-192KB/SM | ~30 cycles | 1.5 TB/s | Automatic caching |
| L2 Cache | 6-72MB | ~200 cycles | 4 TB/s | All memory access |
| Constant | 64KB | ~30 cycles* | High | Read-only data |
| Global | 8-80GB | ~400 cycles | 1 TB/s | Large datasets |

*Cached constant memory access

### 2.2 Constant Memory for Precomputed Tables

```cuda
// Precomputed generator points (constant memory)
__device__ __constant__ uint64_t Gx[512][4];  // G, 2G, 3G, ..., 511G
__device__ __constant__ uint64_t Gy[512][4];

// SHA256 round constants
__device__ __constant__ uint32_t K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    // ... (64 constants)
};
```

Benefits:
- Single memory read broadcast to all threads in warp
- Automatic caching
- No bank conflicts

### 2.3 Register Allocation

```cuda
// 256-bit integer in 4 registers
uint64_t x[4];  // Uses 4 × 64-bit = 4 registers

// EC Point in Jacobian coordinates
uint64_t X[4], Y[4], Z[4];  // 12 registers per point

// Per-thread register budget: ~255 registers
// Optimal: Keep critical data in registers
```

### 2.4 Coalesced Memory Access

```cuda
// Good: Coalesced access (thread i accesses element i)
__device__ void Load256A(uint64_t *r, uint64_t *a) {
    r[0] = a[threadIdx.x];
    r[1] = a[threadIdx.x + blockDim.x];
    r[2] = a[threadIdx.x + 2*blockDim.x];
    r[3] = a[threadIdx.x + 3*blockDim.x];
}

// Bad: Strided access (causes multiple transactions)
__device__ void Load256Bad(uint64_t *r, uint64_t *a) {
    r[0] = a[threadIdx.x * 4];     // Strided!
    r[1] = a[threadIdx.x * 4 + 1];
    // ...
}
```

### 2.5 L1/L2 Cache Configuration

```cpp
// Prefer L1 cache over shared memory
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

// Per-kernel configuration
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferL1);
```

---

## 3. Kernel Optimization Techniques

### 3.1 Occupancy Optimization

```cpp
// Query optimal block size
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                    comp_keys, 0, 0);

// Calculate occupancy
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                               comp_keys,
                                               blockSize, 0);
float occupancy = (float)(numBlocks * blockSize) / maxThreadsPerSM;
```

### 3.2 Warp-Level Optimization

```cuda
// All threads in warp execute same instruction
// Branch divergence kills performance

// Bad: Divergent branch
if (threadIdx.x < 16) {
    // Only half the warp executes
} else {
    // Other half executes
}

// Good: Warp-uniform branch
if (warpIdx < 4) {
    // All threads in warp take same path
}
```

### 3.3 Loop Unrolling

```cuda
// Manual unrolling for known iteration count
#pragma unroll
for (int i = 0; i < 8; i++) {
    s[i] = __byte_perm(s[i], 0, 0x0123);
}

// Partial unrolling for large loops
#pragma unroll 16
for (int i = 0; i < 64; i++) {
    // SHA256 rounds
}
```

### 3.4 Instruction-Level Parallelism

```cuda
// Schedule independent operations together
uint64_t a = x[0] * y[0];  // MUL
uint64_t b = x[1] * y[1];  // MUL (independent, can run parallel)
uint64_t c = x[2] * y[2];  // MUL (independent)

// Dependent chain limits ILP
uint64_t a = x[0] * y[0];
uint64_t b = a + z[0];     // Depends on a
uint64_t c = b * y[1];     // Depends on b
```

### 3.5 Synchronization Minimization

```cuda
// Only sync when necessary
__shared__ uint32_t shared_data[256];

// Thread writes
shared_data[threadIdx.x] = result;

// Sync only when reading others' data
__syncthreads();

// Now safe to read
uint32_t neighbor = shared_data[(threadIdx.x + 1) % 256];
```

---

## 4. 256-bit Arithmetic on GPU

### 4.1 PTX Assembly Intrinsics

```cuda
// Addition with carry chain
#define UADDO(c, a, b) asm("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))
#define UADDC(c, a, b) asm("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))
#define UADD(c, a, b)  asm("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))

// Usage: 256-bit addition
void Add256(uint64_t *r, uint64_t *a, uint64_t *b) {
    UADDO(r[0], a[0], b[0]);  // Add with carry out
    UADDC(r[1], a[1], b[1]);  // Add with carry in/out
    UADDC(r[2], a[2], b[2]);
    UADD(r[3], a[3], b[3]);   // Final add with carry in
}
```

### 4.2 Multiplication Primitives

```cuda
// 64-bit × 64-bit = 128-bit
#define UMULLO(lo, a, b) asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b))
#define UMULHI(hi, a, b) asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b))

// Multiply-accumulate with carry
#define MADDO(r, a, b, c) asm("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c))
#define MADDC(r, a, b, c) asm("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c))
```

### 4.3 256×256-bit Multiplication

```cuda
// Schoolbook multiplication with secp256k1 reduction
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {
    uint64_t r512[8];
    uint64_t t[5];

    // Phase 1: 256×256 = 512-bit product
    UMult(r512, a, b[0]);
    for (int i = 1; i < 4; i++) {
        UMult(t, a, b[i]);
        // Add with carries...
    }

    // Phase 2: Reduce 512 → 320 bits
    // Using: 2^256 ≡ 0x1000003D1 (mod p)
    UMult(t, r512 + 4, 0x1000003D1ULL);
    Add(r512, t);

    // Phase 3: Reduce 320 → 256 bits
    // Handle final carry
}
```

### 4.4 Modular Reduction Optimization

```cuda
// secp256k1 specific: p = 2^256 - 0x1000003D1
__device__ void ModReduce(uint64_t *r) {
    uint64_t c = r[4];  // Overflow

    // r = r[0..3] + c * 0x1000003D1
    uint64_t al, ah;
    UMULLO(al, c, 0x1000003D1ULL);
    UMULHI(ah, c, 0x1000003D1ULL);

    UADDO(r[0], r[0], al);
    UADDC(r[1], r[1], ah);
    UADDC(r[2], r[2], 0);
    UADD(r[3], r[3], 0);
}
```

### 4.5 Modular Inversion

```cuda
// Binary GCD with 62-bit steps
__device__ void _ModInv(uint64_t *R) {
    uint64_t u[5] = {P0, P1, P2, P3, 0};  // p
    uint64_t v[5];
    Load(v, R);

    uint64_t r[5] = {0, 0, 0, 0, 0};
    uint64_t s[5] = {1, 0, 0, 0, 0};

    while (!IsZero(v)) {
        DivStep62(u, v, &uu, &uv, &vu, &vv);
        MatrixVecMul(u, v, uu, uv, vu, vv);
        MatrixVecMul(r, s, uu, uv, vu, vv);
        ShiftR62(u);
        ShiftR62(v);
        ShiftR62(r);
        ShiftR62(s);
    }

    // r now contains inverse
    Load(R, r);
}
```

---

## 5. Performance Profiling

### 5.1 nvprof / Nsight Compute

```bash
# Basic profiling
nvprof ./vanitysearch -gpuId 0 -start ... -range ...

# Detailed metrics
nv-nsight-cu-cli --metrics sm__throughput.avg_pct,\
    dram__throughput.avg_pct,\
    gpu__compute_memory_throughput.avg.pct \
    ./vanitysearch ...
```

### 5.2 Key Metrics to Monitor

| Metric | Target | Meaning |
|--------|--------|---------|
| SM Throughput | >80% | Compute utilization |
| Memory Throughput | >60% | Memory bandwidth usage |
| Achieved Occupancy | >50% | Active warps ratio |
| Warp Execution Efficiency | >90% | Divergence measure |
| Global Load Efficiency | >80% | Coalescing quality |

### 5.3 Register Pressure Analysis

```bash
# Show register usage during compilation
nvcc --ptxas-options=-v ...

# Output example:
# ptxas info: Used 64 registers, 352 bytes smem, ...
```

Target: <64 registers per thread for good occupancy.

### 5.4 Performance Counters

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);
```

---

## 6. Architecture-Specific Tuning

### 6.1 Pascal (GTX 10xx, sm_60/61)

```
Characteristics:
- 64 CUDA cores per SM
- 256KB register file per SM
- 96KB shared memory per SM

Tuning:
- Block size: 256 threads
- Target 50%+ occupancy
- Prefer registers over shared memory
```

### 6.2 Turing (RTX 20xx, sm_75)

```
Characteristics:
- 64 CUDA cores per SM
- Independent integer datapath
- Tensor cores (unused for ECC)

Tuning:
- Leverage independent INT32 unit
- Block size: 256 threads
- Use async copy where applicable
```

### 6.3 Ampere (RTX 30xx, sm_86)

```
Characteristics:
- 128 CUDA cores per SM
- Larger L1/shared memory
- Third-gen Tensor Cores

Tuning:
- Block size: 256-512 threads
- Larger cache = less memory pressure
- Consider larger grid sizes
```

### 6.4 Ada Lovelace (RTX 40xx, sm_89)

```
Characteristics:
- 128 CUDA cores per SM
- 72MB L2 cache
- Higher clock speeds

Tuning:
- Leverage massive L2 cache
- More blocks per SM possible
- Block size: 256 threads
- Grid size: numSMs × 128-256
```

### 6.5 Compilation Flags

```makefile
# Architecture-specific builds
CUDA_ARCH_86 = -gencode=arch=compute_86,code=sm_86
CUDA_ARCH_89 = -gencode=arch=compute_89,code=sm_89

# Optimization flags
NVCC_FLAGS = -O3 \
             --use_fast_math \
             -maxrregcount=0 \
             --ptxas-options=-v

# Single architecture for maximum optimization
ifdef ARCH
    CUDA_ARCH = -gencode=arch=compute_$(ARCH),code=sm_$(ARCH)
endif
```

---

## Summary

Key optimization strategies for GPU ECC:

1. **Memory**: Use constant memory for precomputed tables, ensure coalesced global access
2. **Registers**: Keep hot data in registers, minimize spills
3. **Arithmetic**: Use PTX intrinsics for carry chains, optimize reduction
4. **Parallelism**: Maximize occupancy, minimize divergence
5. **Architecture**: Tune block/grid sizes per GPU generation

Typical performance bottlenecks:
- Modular inversion (solved by batching)
- Memory latency (solved by caching)
- Register pressure (solved by careful algorithm design)

---

*For implementation details, see GPUEngine.cu and GPUMath.h*
