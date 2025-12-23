# VanitySearch-Bitcrack: A Technical Analysis of GPU-Accelerated Bitcoin Key Search

**Technical Research Paper**

*Version 2.2 | December 2024*

## Abstract

This paper presents a comprehensive technical analysis of VanitySearch-Bitcrack, a high-performance GPU-accelerated tool for Bitcoin private key search operations. We examine the cryptographic foundations, algorithmic optimizations, and GPU parallelization strategies employed to achieve throughput exceeding 8 billion key operations per second on modern NVIDIA hardware. The implementation leverages advanced elliptic curve cryptography techniques, batch modular inversion, and optimized CUDA kernels to maximize computational efficiency.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Cryptographic Foundations](#2-cryptographic-foundations)
3. [Algorithm Design](#3-algorithm-design)
4. [GPU Architecture and Optimization](#4-gpu-architecture-and-optimization)
5. [Performance Analysis](#5-performance-analysis)
6. [Security Considerations](#6-security-considerations)
7. [Future Directions](#7-future-directions)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Background

Bitcoin's security model relies on the computational infeasibility of deriving private keys from public addresses. The address derivation process involves:

1. **Private Key Generation**: A 256-bit random integer in the range [1, n-1] where n is the curve order
2. **Public Key Computation**: Elliptic curve point multiplication P = k * G
3. **Address Derivation**: Hash160(PublicKey) followed by Base58Check or Bech32 encoding

VanitySearch-Bitcrack is designed to efficiently search through keyspaces for:
- Known addresses within specified private key ranges (puzzle challenges)
- Vanity addresses with custom prefixes
- Cryptographic research and education

### 1.2 Problem Statement

Given:
- A target Bitcoin address A
- A search range [k_start, k_end] where k_end - k_start = 2^n

Find:
- The private key k such that Address(k * G) = A

The computational complexity is O(2^n) key derivations. This paper analyzes techniques to maximize throughput while maintaining cryptographic correctness.

### 1.3 Contributions

1. Optimized batch modular inversion using Montgomery's trick
2. GPU-parallelized elliptic curve group operations
3. Efficient Hash160 computation with SHA256+RIPEMD160 fusion
4. Memory-efficient address lookup with Bloom filter-like structures

---

## 2. Cryptographic Foundations

### 2.1 SECP256K1 Elliptic Curve

Bitcoin uses the secp256k1 Koblitz curve defined over the prime field F_p:

```
Curve Equation: y^2 = x^3 + 7 (mod p)

Parameters:
  p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    = 2^256 - 2^32 - 977

  n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    (Curve order)

  G = (Gx, Gy) (Generator point)
  Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
  Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
```

### 2.2 Key Properties for Optimization

**Special Prime Structure**: p = 2^256 - 2^32 - 977

This allows efficient modular reduction:
```
r = a mod p
  = (a mod 2^256) + (a >> 256) * (2^32 + 977)
```

The implementation exploits this with the constant:
```c
#define SECP256K1_PRIME_LOW 0x1000003D1ULL  // = 2^32 + 977
```

**Curve a = 0**: The secp256k1 curve has a = 0 in the Weierstrass form, eliminating multiplication by 'a' in point doubling operations.

### 2.3 Point Representation

Points are stored in Jacobian coordinates (X:Y:Z) where:
```
x = X/Z^2
y = Y/Z^3
```

This avoids expensive modular inversions during intermediate calculations, deferring them to a single batch inversion at the end.

### 2.4 Hash Functions

**SHA-256**: Used for the first hash of the public key
- Merkle-Damgard construction
- 64 rounds with 32-bit word operations
- Initial hash values derived from prime square roots

**RIPEMD-160**: Applied to SHA-256 output
- Parallel message processing streams
- 80 rounds (5 groups of 16)
- Produces the 20-byte Hash160

**Hash160 = RIPEMD160(SHA256(PublicKey))**

---

## 3. Algorithm Design

### 3.1 Group-Based Key Generation

Instead of computing each public key independently, we use the group method:

Given starting key k and group size G:
```
P[0] = k * G           (initial point)
P[i] = P[0] + i * G    (for i = 1 to G-1)
```

Since i * G can be precomputed, we only need point additions:
```
P[i] = P[0] + Precomputed[i]
```

### 3.2 Batch Modular Inversion

Converting from Jacobian to affine coordinates requires modular inversion. Using Montgomery's trick, we compute n inversions with only 1 actual inversion:

```
Input: x[0], x[1], ..., x[n-1]

// Forward pass: compute products
q[0] = x[0]
for i = 1 to n-1:
    q[i] = q[i-1] * x[i]

// Single inversion
inv = ModInv(q[n-1])

// Backward pass: compute individual inverses
for i = n-1 down to 1:
    x[i]^(-1) = q[i-1] * inv
    inv = inv * x[i]
x[0]^(-1) = inv
```

This reduces O(n) inversions to O(n) multiplications + 1 inversion.

### 3.3 Modular Inversion Algorithm

The implementation uses a variant of the Binary Extended GCD algorithm optimized for the secp256k1 prime:

```
DivStep62 Algorithm:
- Process 62 bits per iteration (optimal for 64-bit operations)
- Use signed integer arithmetic for bidirectional search
- Terminate when remainder reaches 1
```

### 3.4 Address Matching Strategy

**Two-Level Lookup Table**:

1. **Level 1**: 16-bit prefix lookup (65536 entries)
   - Fast O(1) access
   - Filters 99.998% of non-matching addresses

2. **Level 2**: 32-bit secondary verification
   - Full address comparison only when L1 matches
   - Handles collision resolution

### 3.5 Endomorphism Optimization (Optional)

secp256k1 has an efficiently computable endomorphism:
```
lambda * (x, y) = (beta * x, y)

where:
  beta^3 = 1 (mod p)
  lambda^3 = 1 (mod n)
```

This allows computing 3 related points from 1 scalar multiplication, tripling effective throughput for certain search patterns.

---

## 4. GPU Architecture and Optimization

### 4.1 CUDA Programming Model

**Thread Hierarchy**:
```
Grid -> Blocks -> Warps -> Threads

Configuration:
- 256 threads per block (NB_THREAD_PER_GROUP)
- Multiple blocks per SM
- 32 threads per warp (lockstep execution)
```

### 4.2 Memory Hierarchy Optimization

**Constant Memory (64KB)**:
- Precomputed generator point table (Gx[], Gy[])
- SHA256 round constants K[]
- RIPEMD160 constants K160[]

**Register Usage**:
- 256-bit integers stored in 4x64-bit registers
- Intermediate EC points: 12 registers (3 coordinates x 4 words)
- Hash state: 8-16 registers

**L1/L2 Cache Configuration**:
```c
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
```

### 4.3 256-bit Modular Arithmetic on GPU

**Addition with Carry Chain**:
```cuda
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));
```

**Montgomery Multiplication**:
- Multiply-accumulate with carry propagation
- Reduction using secp256k1 special form
- Full 512-bit intermediate result reduced to 256 bits

**Modular Squaring**:
Specialized algorithm exploiting symmetry:
```
a^2 = sum of a[i]*a[j] for all i,j
    = sum(a[i]^2) + 2*sum(a[i]*a[j] for i<j)
```

### 4.4 Hash Function Optimization

**SHA256 Optimizations**:
- Message schedule computed inline
- Unrolled rounds
- Compiler-friendly macro structure

**RIPEMD160 Optimizations**:
- Parallel left/right branches
- Optimized rotation using PTX instructions
- Combined state update

### 4.5 Kernel Structure

```cuda
__global__ void comp_keys(
    address_t* sAddress,    // Level 1 lookup table
    uint32_t* lookup32,     // Level 2 lookup table
    uint64_t* keys,         // Starting points (X,Y coordinates)
    uint32_t* out           // Output buffer for matches
) {
    // 1. Load starting point for this thread
    // 2. Check starting point
    // 3. For each offset in group:
    //    a. Compute new point using EC addition
    //    b. Derive compressed public key
    //    c. Compute Hash160
    //    d. Check against lookup tables
    // 4. Update starting point for next iteration
}
```

### 4.6 Memory Access Patterns

**Coalesced Access**: Thread i accesses memory location base + i
```cuda
#define Load256A(r, a) {\
  (r)[0] = (a)[IDX]; \
  (r)[1] = (a)[IDX+blockDim.x]; \
  (r)[2] = (a)[IDX+2*blockDim.x]; \
  (r)[3] = (a)[IDX+3*blockDim.x];}
```

This ensures 128-byte aligned accesses for maximum memory bandwidth.

---

## 5. Performance Analysis

### 5.1 Benchmark Results

| GPU | Architecture | Performance (MKey/s) | Power (W) | Efficiency (MKey/J) |
|-----|-------------|---------------------|-----------|---------------------|
| RTX 5090 | Blackwell | 8,800 | 450 | 19.6 |
| RTX 4090 | Ada Lovelace | 6,900 | 450 | 15.3 |
| RTX 4080 | Ada Lovelace | 4,800 | 320 | 15.0 |
| RTX 3090 | Ampere | 4,500 | 350 | 12.9 |
| RTX 3080 | Ampere | 3,800 | 320 | 11.9 |
| RTX 2080 Ti | Turing | 2,500 | 250 | 10.0 |
| V100 | Volta | 3,200 | 300 | 10.7 |

### 5.2 Operation Breakdown

Approximate cost per key check:
```
EC Point Addition:     ~40%
Hash160 Computation:   ~35%
Address Lookup:        ~5%
Memory Operations:     ~15%
Other:                 ~5%
```

### 5.3 Scalability Analysis

**Strong Scaling**: Performance scales linearly with GPU count for independent range searches.

**Weak Scaling**: Memory usage is O(n) where n is the number of target addresses.

### 5.4 Theoretical Limits

For secp256k1 with batch processing:
- EC addition: ~8 modular multiplications
- Modular multiplication: ~16 64-bit multiply-accumulates
- SHA256: 64 rounds, ~2000 operations
- RIPEMD160: 160 rounds, ~2500 operations

Theoretical peak (RTX 4090):
- 82.6 TFLOPS (FP32)
- Estimated: ~10 GKey/s (achievable: 6.9 GKey/s = 66% efficiency)

---

## 6. Security Considerations

### 6.1 Cryptographic Security

The security of Bitcoin addresses relies on:
1. **ECDLP Hardness**: Computing k from k*G is computationally infeasible
2. **Hash Function Security**: Collision/preimage resistance of SHA256 and RIPEMD160
3. **Keyspace Size**: 2^256 possible private keys

### 6.2 Search Space Analysis

For a random n-bit search:
- Expected keys to find target: 2^(n-1)
- At 6.9 GKey/s (RTX 4090):
  - 40-bit range: ~80 seconds
  - 50-bit range: ~22 hours
  - 60-bit range: ~2.5 years
  - 70-bit range: ~2,600 years

### 6.3 Ethical Considerations

This tool should only be used for:
- Bitcoin puzzle challenges (educational)
- Vanity address generation (personal use)
- Cryptographic research
- Security auditing with authorization

Attempting to recover private keys from arbitrary addresses is:
1. Computationally infeasible for secure keys
2. Potentially illegal in many jurisdictions
3. Ethically problematic

### 6.4 Random Number Generation

The random mode uses:
- System entropy sources (time, process ID)
- Mersenne Twister for uniform distribution
- Key space partitioning for thread independence

---

## 7. Future Directions

### 7.1 Algorithmic Improvements

1. **Pollard's Rho**: For certain attack scenarios, could reduce complexity from O(n) to O(sqrt(n))
2. **Rainbow Tables**: Precomputed address-to-key mappings for common patterns
3. **Quantum Considerations**: Post-quantum signature schemes for future Bitcoin upgrades

### 7.2 Hardware Optimizations

1. **Tensor Cores**: Investigating use of matrix operations for batch EC arithmetic
2. **Multi-GPU Scaling**: Improved work distribution and result aggregation
3. **FPGA/ASIC**: Custom hardware for maximum efficiency

### 7.3 Software Improvements

1. **Dynamic Load Balancing**: Runtime adjustment of work distribution
2. **Checkpoint/Resume**: Robust state saving for long-running searches
3. **Network Distribution**: Coordinated search across multiple machines

---

## 8. References

1. **Bitcoin Whitepaper**: Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.

2. **SECP256K1 Standard**: Certicom Research. (2000). SEC 2: Recommended Elliptic Curve Domain Parameters.

3. **SHA-256 Specification**: NIST FIPS 180-4 (2015). Secure Hash Standard.

4. **RIPEMD-160**: Dobbertin, H., Bosselaers, A., & Preneel, B. (1996). RIPEMD-160: A Strengthened Version of RIPEMD.

5. **Montgomery Multiplication**: Montgomery, P.L. (1985). Modular Multiplication Without Trial Division.

6. **CUDA Programming Guide**: NVIDIA Corporation. (2024). CUDA C++ Programming Guide.

7. **VanitySearch Original**: Pons, J.L. (2019). VanitySearch: A Bitcoin Vanity Address Generator.

8. **Batch Inversion**: Montgomery, P.L. (1987). Speeding the Pollard and Elliptic Curve Methods of Factorization.

---

## Appendix A: Performance Tuning Guide

### A.1 Optimal Configuration

```bash
# For RTX 4090
./vanitysearch -gpuId 0 -m 524288 -start <HEX> -range <BITS> <ADDRESS>

# Parameters:
#   -m 524288: Increased max found (8 * 65536)
#   Use multiples of 65536 for -m parameter
```

### A.2 Memory Requirements

| Addresses | RAM Usage | GPU VRAM |
|-----------|-----------|----------|
| 1 | ~50 MB | ~100 MB |
| 1000 | ~55 MB | ~110 MB |
| 100000 | ~100 MB | ~200 MB |
| 1000000 | ~500 MB | ~600 MB |

---

## Appendix B: Glossary

- **ECDLP**: Elliptic Curve Discrete Logarithm Problem
- **Hash160**: RIPEMD160(SHA256(x))
- **Jacobian Coordinates**: Projective representation (X:Y:Z) where x=X/Z^2, y=Y/Z^3
- **Modular Inversion**: Finding a^(-1) such that a * a^(-1) = 1 (mod p)
- **secp256k1**: The specific elliptic curve used by Bitcoin
- **Warp**: 32 GPU threads executing in lockstep

---

*Document Version: 2.2*
*Last Updated: December 2024*
*License: GPL v3.0*
