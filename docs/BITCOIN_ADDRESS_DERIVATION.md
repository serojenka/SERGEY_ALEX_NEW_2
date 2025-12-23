# Bitcoin Address Derivation

**Complete Guide to Bitcoin Address Generation**

## Table of Contents

1. [Overview](#1-overview)
2. [Private Key Generation](#2-private-key-generation)
3. [Public Key Computation](#3-public-key-computation)
4. [Address Types](#4-address-types)
5. [Hash Functions](#5-hash-functions)
6. [Encoding Schemes](#6-encoding-schemes)
7. [Implementation Details](#7-implementation-details)

---

## 1. Overview

### 1.1 Address Derivation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIVATE KEY (256 bits)                       │
│                 Random integer in [1, n-1]                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ELLIPTIC CURVE MULTIPLICATION                       │
│                    P = k × G                                     │
│              (secp256k1 point multiplication)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PUBLIC KEY                                    │
│   Uncompressed: 04 || X (32 bytes) || Y (32 bytes)  [65 bytes]  │
│   Compressed:   02/03 || X (32 bytes)               [33 bytes]  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        HASH160                                   │
│              RIPEMD160(SHA256(public_key))                       │
│                       [20 bytes]                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌────────┐   ┌────────┐   ┌────────────┐
         │ P2PKH  │   │  P2SH  │   │   Bech32   │
         │  (1..) │   │  (3..) │   │   (bc1..)  │
         └────────┘   └────────┘   └────────────┘
```

### 1.2 Address Types Summary

| Type | Prefix | Format | Size | Introduced |
|------|--------|--------|------|------------|
| P2PKH | 1 | Base58Check | 25-34 chars | 2009 |
| P2SH | 3 | Base58Check | 34 chars | BIP 16 (2012) |
| P2WPKH | bc1q | Bech32 | 42 chars | BIP 84 (2017) |
| P2TR | bc1p | Bech32m | 62 chars | Taproot (2021) |

---

## 2. Private Key Generation

### 2.1 Key Requirements

A valid Bitcoin private key k must satisfy:
```
1 ≤ k < n

where n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
          (secp256k1 curve order)
```

### 2.2 Key Space Size

```
Valid keys: n - 1 ≈ 1.158 × 10^77

In bits: log₂(n) ≈ 256 bits

Comparison:
- Atoms in observable universe: ~10^80
- SHA256 output space: 2^256 ≈ 1.16 × 10^77
```

### 2.3 Key Representations

**Raw Hexadecimal (64 characters)**:
```
E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262
```

**WIF (Wallet Import Format)**:
```
Uncompressed: 5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ
Compressed:   KxFC1jmwwCoACiCAWZ3eXa96mBM6tb3TYzGmf6YwgdGWZgawvrtJ
```

**WIF Structure**:
```
┌────────┬──────────────────┬────────┬──────────┐
│ Prefix │   Private Key    │ Suffix │ Checksum │
│ (1B)   │    (32 bytes)    │ (0/1B) │ (4 bytes)│
└────────┴──────────────────┴────────┴──────────┘

Prefix: 0x80 (mainnet), 0xEF (testnet)
Suffix: 0x01 if compressed, absent if uncompressed
Checksum: First 4 bytes of SHA256(SHA256(prefix || key || suffix))
```

---

## 3. Public Key Computation

### 3.1 Point Multiplication

```
P = k × G

where:
  k = private key (256-bit integer)
  G = generator point
  P = (Px, Py) = public key point
```

### 3.2 Uncompressed Public Key

```
Format: 04 || Px || Py

Size: 1 + 32 + 32 = 65 bytes

Example:
04
79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
```

### 3.3 Compressed Public Key

Since y² = x³ + 7 (mod p), knowing x determines y up to sign.

```
Format: prefix || Px

Prefix: 02 if Py is even, 03 if Py is odd
Size: 1 + 32 = 33 bytes

Example (even Y):
0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798

Example (odd Y):
0379BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
```

### 3.4 Y-Coordinate Recovery

Given x and parity bit:
```
y² = x³ + 7 (mod p)
y = (x³ + 7)^((p+1)/4) (mod p)    // p ≡ 3 (mod 4)

if parity(y) ≠ expected:
    y = p - y
```

---

## 4. Address Types

### 4.1 P2PKH (Pay to Public Key Hash)

**Prefix**: 1 (mainnet), m/n (testnet)

**Script**: OP_DUP OP_HASH160 <hash160> OP_EQUALVERIFY OP_CHECKSIG

**Derivation**:
```
1. Hash160 = RIPEMD160(SHA256(compressed_pubkey))
2. Payload = 0x00 || Hash160
3. Checksum = SHA256(SHA256(Payload))[0:4]
4. Address = Base58(Payload || Checksum)
```

**Example**:
```
Hash160:    751E76E8199196D454941C45D1B3A323F1433BD6
Payload:    00751E76E8199196D454941C45D1B3A323F1433BD6
Checksum:   54D35A12
Full:       00751E76E8199196D454941C45D1B3A323F1433BD654D35A12
Address:    1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
```

### 4.2 P2SH (Pay to Script Hash)

**Prefix**: 3 (mainnet), 2 (testnet)

**Script**: OP_HASH160 <script_hash> OP_EQUAL

For P2WPKH-P2SH (nested SegWit):
```
1. RedeemScript = OP_0 || OP_PUSH20 || Hash160
   = 0x00 || 0x14 || Hash160
2. ScriptHash = RIPEMD160(SHA256(RedeemScript))
3. Payload = 0x05 || ScriptHash
4. Address = Base58Check(Payload)
```

**Example**:
```
Hash160:        751E76E8199196D454941C45D1B3A323F1433BD6
RedeemScript:   0014751E76E8199196D454941C45D1B3A323F1433BD6
ScriptHash:     BCFEB728B584253D5F3F70BCB780E9EF218A68F4
Payload:        05BCFEB728B584253D5F3F70BCB780E9EF218A68F4
Address:        3JvL6Ymt8MVWiCNHC7oWU6nLeHNJKLZGLN
```

### 4.3 P2WPKH (Native SegWit - Bech32)

**Prefix**: bc1q (mainnet), tb1q (testnet)

**Witness Program**: OP_0 <hash160>

**Derivation**:
```
1. Hash160 = RIPEMD160(SHA256(compressed_pubkey))
2. Witness version = 0
3. Address = Bech32Encode("bc", 0, Hash160)
```

**Example**:
```
Hash160: 751E76E8199196D454941C45D1B3A323F1433BD6
Address: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
```

### 4.4 Comparison

| Feature | P2PKH | P2SH | P2WPKH |
|---------|-------|------|--------|
| Size | 25 bytes | 23 bytes | 22 bytes |
| Fees | Highest | Medium | Lowest |
| Error detection | 4 bytes | 4 bytes | 6 chars |
| Case sensitive | Yes | Yes | No |

---

## 5. Hash Functions

### 5.1 SHA-256

**Input**: Message of arbitrary length
**Output**: 256 bits (32 bytes)

**Algorithm**:
```
1. Pad message to multiple of 512 bits
2. Initialize 8 state variables (H0-H7)
3. Process each 512-bit block:
   a. Prepare 64-word message schedule
   b. 64 rounds of compression
   c. Add compressed chunk to current hash
4. Produce final 256-bit hash
```

**Round function**:
```
T1 = h + Σ1(e) + Ch(e,f,g) + Kt + Wt
T2 = Σ0(a) + Maj(a,b,c)
h = g; g = f; f = e; e = d + T1
d = c; c = b; b = a; a = T1 + T2
```

### 5.2 RIPEMD-160

**Input**: Message of arbitrary length
**Output**: 160 bits (20 bytes)

**Structure**:
- Two parallel computation streams (left and right)
- 80 rounds each (5 groups of 16)
- Different permutations and rotations per stream
- Combined at end

**Benefits**:
- Developed independently from NSA-designed hashes
- Different structure provides defense in depth
- Shorter output reduces address length

### 5.3 Hash160

**Definition**: Hash160(x) = RIPEMD160(SHA256(x))

**Purpose**:
- Compress public key to 160 bits
- Provide quantum resistance margin (160-bit vs 256-bit preimage)
- Balance security and address length

---

## 6. Encoding Schemes

### 6.1 Base58Check

**Alphabet**: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
- Excludes: 0, O, I, l (ambiguous characters)

**Encoding**:
```
input: byte array
1. checksum = SHA256(SHA256(input))[0:4]
2. data = input || checksum
3. Convert data to big integer
4. Repeatedly divide by 58, collect remainders
5. Map remainders to Base58 alphabet
6. Prepend '1' for each leading zero byte
```

**Decoding**:
```
1. Map characters to values
2. Compute big integer from base-58 representation
3. Convert to bytes
4. Verify checksum
5. Return payload without checksum
```

### 6.2 Bech32

**Alphabet**: qpzry9x8gf2tvdw0s3jn54khce6mua7l
- All lowercase
- No similar-looking characters

**Structure**:
```
┌──────┬───┬────────────────────────────────────┬────────┐
│ HRP  │ 1 │            Data                    │Checksum│
│ (bc) │   │ (witness version + converted data) │(6 char)│
└──────┴───┴────────────────────────────────────┴────────┘
```

**Data encoding** (8-to-5 bit conversion):
```
Input:  20 bytes (160 bits)
Output: 32 quintet (5-bit values)

Process: Pack 8 bits, unpack 5 bits at a time
```

**Checksum** (BCH code):
- Detects up to 4 errors
- Always detects single-character substitution
- Uses polynomial: x^6 + x^4 + x^2 + x + 1

---

## 7. Implementation Details

### 7.1 Compressed Public Key Creation

```cpp
void GetCompressedPubKey(Point &p, uint8_t *out) {
    // Prefix: 02 if y is even, 03 if y is odd
    out[0] = p.y.IsEven() ? 0x02 : 0x03;

    // Copy x coordinate (32 bytes)
    p.x.Get32Bytes(out + 1);
}
```

### 7.2 Hash160 Computation

```cpp
void GetHash160(uint8_t *pubKey, int len, uint8_t *hash) {
    uint8_t sha256_result[32];

    // SHA256
    sha256(pubKey, len, sha256_result);

    // RIPEMD160
    ripemd160_32(sha256_result, hash);
}
```

### 7.3 P2PKH Address Generation

```cpp
std::string GetP2PKHAddress(uint8_t *hash160) {
    uint8_t payload[25];

    // Version byte
    payload[0] = 0x00;

    // Hash160
    memcpy(payload + 1, hash160, 20);

    // Checksum
    sha256_checksum(payload, 21, payload + 21);

    // Base58 encode
    return EncodeBase58(payload, payload + 25);
}
```

### 7.4 Bech32 Address Generation

```cpp
std::string GetBech32Address(uint8_t *hash160) {
    char output[128];

    // Witness version 0, 20-byte program
    segwit_addr_encode(output, "bc", 0, hash160, 20);

    return std::string(output);
}
```

### 7.5 GPU-Optimized Hash160

```cuda
__device__ void _GetHash160Comp(uint64_t *x, uint8_t isOdd, uint8_t *hash) {
    uint32_t *x32 = (uint32_t *)x;
    uint32_t pubKey[16];
    uint32_t sha256_state[16];

    // Build compressed public key (33 bytes)
    pubKey[0] = __byte_perm(x32[7], 0x02 + isOdd, 0x4321);
    pubKey[1] = __byte_perm(x32[7], x32[6], 0x0765);
    // ... (continue for all 8 words)
    pubKey[8] = __byte_perm(x32[0], 0x80, 0x0456);
    // ... (padding)
    pubKey[15] = 0x108;  // Length in bits (33 * 8 = 264)

    // SHA256
    SHA256Transform(sha256_state, pubKey);

    // Prepare for RIPEMD160
    for (int i = 0; i < 8; i++)
        sha256_state[i] = __byte_perm(sha256_state[i], 0, 0x0123);

    // RIPEMD160
    RIPEMD160Initialize((uint32_t *)hash);
    RIPEMD160Transform((uint32_t *)hash, sha256_state);
}
```

---

## Summary

Bitcoin address derivation involves:

1. **Private Key**: 256-bit random integer in valid range
2. **Public Key**: Elliptic curve point multiplication
3. **Hash160**: SHA256 followed by RIPEMD160
4. **Encoding**: Base58Check or Bech32 depending on address type

Security considerations:
- Private key must be truly random
- Compressed public keys are preferred (smaller, same security)
- Bech32 addresses have better error detection
- Hash160 provides defense against potential SHA256 weaknesses

---

*For more details, see SECP256K1_MATHEMATICS.md and TECHNICAL_PAPER.md*
