# VanitySearch-Bitcrack

**High-Performance GPU-Accelerated Bitcoin Private Key Search Tool**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)

An optimized fork of [VanitySearch](https://github.com/JeanLucPons/VanitySearch) by Jean Luc PONS, specifically tailored for Bitcoin puzzle challenges with significant performance improvements.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance Benchmarks](#performance-benchmarks)
- [Installation](#installation)
- [Usage](#usage)
- [Search Modes](#search-modes)
- [Command-Line Options](#command-line-options)
- [Examples](#examples)
- [Technical Documentation](#technical-documentation)
- [Architecture](#architecture)
- [Building from Source](#building-from-source)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

VanitySearch-Bitcrack is a specialized tool for searching Bitcoin private keys within defined ranges. It leverages NVIDIA CUDA technology to achieve massive parallelization, enabling billions of key checks per second. The tool is particularly useful for:

- **Bitcoin Puzzle Challenges**: Searching for known addresses within specified key ranges
- **Vanity Address Generation**: Finding addresses with custom prefixes (e.g., "1Bitcoin...")
- **Cryptographic Research**: Studying elliptic curve operations at scale

### How It Works

1. **Key Generation**: Generates Bitcoin private keys within a specified hexadecimal range
2. **Public Key Derivation**: Computes the corresponding public key using SECP256K1 elliptic curve multiplication
3. **Address Computation**: Derives Bitcoin addresses using SHA256 + RIPEMD160 hashing with Base58Check or Bech32 encoding
4. **Pattern Matching**: Compares generated addresses against target addresses or prefixes
5. **Result Output**: Reports matching private keys when found

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **CUDA Optimization** | Highly optimized modular arithmetic using PTX assembly |
| **Memory Efficiency** | Minimal RAM usage through optimized data structures |
| **Batch Operations** | ECC addition with batch modular inverse for starting key computation |
| **Power-of-2 Ranges** | Simplified range definition as 2^n bit ranges |
| **Single GPU Focus** | Optimized for single GPU efficiency and control |
| **Compressed Only** | Optimized for compressed address formats |

### Advanced Features

- **Pause/Resume**: Press 'p' to pause GPU operations and free resources, press again to resume
- **Prefix Search**: Wildcard pattern matching for vanity address generation
- **Random Mode**: Each GPU thread scans 1024 consecutive random keys per step
- **Backup Mode**: Automatic checkpoint saving every ~60 seconds for resumable searches

## Performance Benchmarks

| GPU Model | Performance (MKeys/s) | Architecture |
|-----------|----------------------|--------------|
| RTX 5090 | 8,800 | Blackwell |
| RTX 4090 | 6,900 | Ada Lovelace |
| RTX 3090 | ~4,500 | Ampere |
| RTX 3080 | ~3,800 | Ampere |
| RTX 2080 Ti | ~2,500 | Turing |

*Benchmarks are approximate and may vary based on driver version, system configuration, and search parameters.*

## Installation

### Pre-built Binaries

Pre-compiled binaries are available in the `VanitySearch 2.2/` directory for Windows and Linux.

### System Requirements

- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- CUDA Toolkit 11.0 or later
- Linux: GCC 9+ with SSE3 support
- Windows: Visual Studio 2019+ with CUDA integration

## Usage

### Basic Syntax

```bash
./vanitysearch [-v] [-gpuId N] [-i inputfile] [-o outputfile] [-start HEX] [-range N] [-m N] [-stop] [-random] [-backup]
```

### Quick Start

Search for a specific address in a 40-bit range:

```bash
./vanitysearch -gpuId 0 -start 3BA89530000000000 -range 40 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ
```

## Search Modes

### Sequential Mode (Default)

Scans keys sequentially from `start` to `start + 2^range`:

```bash
./vanitysearch -gpuId 0 -start 100000000000000000 -range 68 19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG
```

### Random Mode

Each GPU thread randomly selects positions within the range and scans 1024 consecutive keys (512 forward, 512 backward):

```bash
./vanitysearch -gpuId 0 -start 100000000000000000 -range 68 -random 19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG
```

**Note**: Random mode has no memory of previously scanned keys. As coverage increases, the probability of re-scanning keys also increases.

### Backup/Resume Mode

Enable automatic checkpointing for long-running sequential searches:

```bash
# Start with backup enabled
./vanitysearch -gpuId 0 -start 3BA89530000000000 -range 41 -backup 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ

# Resume from checkpoint (after restart)
./vanitysearch -gpuId 0 -start 3BA89530000000000 -range 41 -backup 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-v` | Print version information | - |
| `-gpuId N` | GPU device ID to use | 0 |
| `-i FILE` | Input file containing addresses/prefixes | - |
| `-o FILE` | Output file for results | stdout |
| `-start HEX` | Starting private key in hexadecimal | Required |
| `-range N` | Bit range dimension (searches start to start + 2^N) | Required |
| `-m N` | Max prefixes per kernel call (multiples of 65536) | 262144 |
| `-stop` | Stop when all prefixes are found | - |
| `-random` | Enable random search mode | Disabled |
| `-backup` | Enable backup/resume mode (sequential only) | Disabled |

### Parameter Guidelines

- **-m Parameter**: When searching for multiple prefixes, increase this value. Use multiples of 65536. Higher values may slightly reduce speed but prevent missed matches.
- **-range Parameter**: Defines the search space as 2^N keys. A range of 40 means searching 2^40 (approximately 1 trillion) keys.

## Examples

### Windows

```bash
# Search with input file
./VanitySearch.exe -gpuId 0 -i input.txt -o output.txt -start 3BA89530000000000 -range 40

# Search single address on GPU 1
./VanitySearch.exe -gpuId 1 -o output.txt -start 3BA89530000000000 -range 42 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ

# Random mode search
./VanitySearch.exe -gpuId 0 -start 100000000000000000 -range 68 -random 19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG

# Sequential with backup
./VanitySearch.exe -gpuId 0 -start 3BA89530000000000 -range 41 -backup 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ
```

### Linux

```bash
# Search with input file
./vanitysearch -gpuId 0 -i input.txt -o output.txt -start 3BA89530000000000 -range 40

# Direct address search
./vanitysearch -gpuId 0 -start 3BA89530000000000 -range 41 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ
```

### Input File Format

When using `-i inputfile`, list one address or prefix per line:

```
1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ
19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG
1Bitcoin*
```

## Technical Documentation

Comprehensive research papers and technical documentation are available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Technical Paper](docs/TECHNICAL_PAPER.md) | Comprehensive cryptographic research paper |
| [SECP256K1 Mathematics](docs/SECP256K1_MATHEMATICS.md) | Mathematical foundations of elliptic curves |
| [GPU Optimization](docs/GPU_OPTIMIZATION.md) | CUDA optimization techniques and strategies |
| [Bitcoin Address Derivation](docs/BITCOIN_ADDRESS_DERIVATION.md) | Complete address generation process |
| [Performance Analysis](docs/PERFORMANCE_ANALYSIS.md) | Benchmarks and optimization analysis |
| [Security Considerations](docs/SECURITY_CONSIDERATIONS.md) | Security implications and best practices |

## Architecture

### Core Components

```
VanitySearch-Bitcrack/
├── Core Cryptography
│   ├── Int.cpp/h           # 256-bit integer arithmetic
│   ├── IntMod.cpp          # Modular arithmetic operations
│   ├── IntGroup.cpp/h      # Batch modular inverse
│   ├── Point.cpp/h         # Elliptic curve point operations
│   └── SECP256K1.cpp/h     # Bitcoin curve implementation
│
├── Encoding
│   ├── Base58.cpp/h        # Legacy address encoding
│   ├── Bech32.cpp/h        # SegWit address encoding
│   └── Wildcard.cpp/h      # Pattern matching
│
├── Hash Functions (hash/)
│   ├── sha256.cpp/h        # SHA-256 (SSE optimized)
│   ├── sha512.cpp/h        # SHA-512 (SSE optimized)
│   └── ripemd160.cpp/h     # RIPEMD-160 (SSE optimized)
│
├── GPU Implementation (GPU/)
│   ├── GPUEngine.cu/h      # CUDA kernel execution
│   ├── GPUMath.h           # GPU modular arithmetic (PTX)
│   ├── GPUGroup.h          # Precomputed point tables
│   ├── GPUHash.h           # GPU hash implementations
│   ├── GPUBase58.h         # GPU Base58 encoding
│   └── GPUWildcard.h       # GPU pattern matching
│
├── Application
│   ├── main.cpp            # Entry point and CLI parsing
│   └── Vanity.cpp/h        # Search orchestration
│
└── docs/                   # Research papers and documentation
```

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION                           │
│  Load target addresses → Initialize GPU → Precompute EC tables │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      KEY GENERATION (GPU)                        │
│  For each thread:                                                │
│    1. Compute starting point: G * private_key                   │
│    2. Iterate through batch using EC point addition             │
│    3. Derive compressed public key (33 bytes)                   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ADDRESS DERIVATION (GPU)                      │
│  1. SHA256(public_key) → 32 bytes                               │
│  2. RIPEMD160(sha256_result) → 20 bytes (Hash160)               │
│  3. Add version byte + Base58Check encode                        │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PATTERN MATCHING (GPU)                       │
│  Compare derived address against target addresses/prefixes      │
│  If match found → Store result in output buffer                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RESULT COLLECTION                           │
│  Copy matches from GPU → Verify on CPU → Output to file/stdout  │
└─────────────────────────────────────────────────────────────────┘
```

## Building from Source

### Linux

```bash
# Install dependencies
sudo apt-get install build-essential g++-9

# Install CUDA Toolkit (version 11.0+)
# Download from: https://developer.nvidia.com/cuda-downloads

# Build
make

# The binary will be created as ./vanitysearch
```

### Makefile Configuration

The Makefile supports multiple GPU architectures:

- `sm_60`: Pascal (GTX 1060, 1070, 1080)
- `sm_61`: Pascal (GTX 1050, 1080 Ti)
- `sm_70`: Volta (V100)
- `sm_75`: Turing (RTX 2060, 2070, 2080)
- `sm_86`: Ampere (RTX 3060, 3070, 3080, 3090)
- `sm_89`: Ada Lovelace (RTX 4060, 4070, 4080, 4090)

### Windows

Use Visual Studio with CUDA integration. Open the solution file and build for Release configuration.

## Contributing

Contributions are welcome! Please ensure:

1. Code follows existing style conventions
2. Changes are tested on supported GPU architectures
3. Documentation is updated for new features

## License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

- **Jean Luc PONS** - Original VanitySearch implementation
- **FixedPaul** - Performance optimizations for Bitcoin puzzle challenges
- **Pieter Wuille** - Bech32 reference implementation
- **Bitcoin Core Developers** - Cryptographic standards and specifications

## Donations

If you find this tool useful, donations are appreciated:

**BTC**: `bc1qag46ashuyatndd05s0aqeq9d6495c29fjezj09`

---

*This software is provided for educational and research purposes. Users are responsible for ensuring compliance with applicable laws and regulations in their jurisdiction.*
