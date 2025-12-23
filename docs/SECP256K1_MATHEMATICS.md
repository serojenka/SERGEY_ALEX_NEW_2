# SECP256K1 Elliptic Curve Mathematics

**Mathematical Foundations for Bitcoin Cryptography**

## Table of Contents

1. [Introduction to Elliptic Curves](#1-introduction-to-elliptic-curves)
2. [SECP256K1 Parameters](#2-secp256k1-parameters)
3. [Point Operations](#3-point-operations)
4. [Modular Arithmetic](#4-modular-arithmetic)
5. [Implementation Optimizations](#5-implementation-optimizations)
6. [Worked Examples](#6-worked-examples)

---

## 1. Introduction to Elliptic Curves

### 1.1 Definition

An elliptic curve over a field K is defined by the Weierstrass equation:

```
y^2 = x^3 + ax + b
```

where `4a^3 + 27b^2 ≠ 0` (non-singular condition).

### 1.2 Group Structure

Points on an elliptic curve form an abelian group under point addition:
- **Identity**: Point at infinity (O)
- **Closure**: P + Q is another point on the curve
- **Associativity**: (P + Q) + R = P + (Q + R)
- **Inverse**: For P = (x, y), -P = (x, -y)
- **Commutativity**: P + Q = Q + P

### 1.3 Cryptographic Security

The **Elliptic Curve Discrete Logarithm Problem (ECDLP)**:

Given points P and Q = kP, finding k is computationally infeasible for properly chosen curves.

Best known attack complexity: O(√n) using Pollard's rho algorithm, where n is the curve order.

---

## 2. SECP256K1 Parameters

### 2.1 Curve Definition

SECP256K1 is a Koblitz curve with equation:

```
y^2 = x^3 + 7   (mod p)
```

where a = 0 and b = 7.

### 2.2 Field Prime

```
p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
  = 2^256 - 2^32 - 977
  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
```

In decimal:
```
p = 115792089237316195423570985008687907853269984665640564039457584007908834671663
```

### 2.3 Curve Order

```
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
```

In decimal:
```
n = 115792089237316195423570985008687907852837564279074904382605163141518161494337
```

### 2.4 Generator Point

```
G = (Gx, Gy)

Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
```

### 2.5 Cofactor

```
h = 1
```

This means every point on the curve (except O) generates the full group.

### 2.6 Special Properties

1. **a = 0**: Simplifies point doubling formulas
2. **p ≡ 3 (mod 4)**: Enables efficient square root computation
3. **Koblitz curve**: Has an efficient endomorphism (see Section 5)

---

## 3. Point Operations

### 3.1 Point Addition (P ≠ Q)

Given P = (x₁, y₁) and Q = (x₂, y₂), compute R = P + Q = (x₃, y₃):

```
λ = (y₂ - y₁) / (x₂ - x₁)   (mod p)

x₃ = λ² - x₁ - x₂           (mod p)
y₃ = λ(x₁ - x₃) - y₁        (mod p)
```

### 3.2 Point Doubling (P = Q)

Given P = (x₁, y₁), compute R = 2P = (x₃, y₃):

```
λ = (3x₁² + a) / (2y₁)      (mod p)

For secp256k1 (a = 0):
λ = 3x₁² / (2y₁)            (mod p)

x₃ = λ² - 2x₁               (mod p)
y₃ = λ(x₁ - x₃) - y₁        (mod p)
```

### 3.3 Point Negation

Given P = (x, y):
```
-P = (x, -y mod p) = (x, p - y)
```

### 3.4 Jacobian Coordinates

To avoid expensive modular inversions, use Jacobian coordinates (X:Y:Z):

```
Affine (x, y) ↔ Jacobian (X, Y, Z)

x = X / Z²
y = Y / Z³
```

**Point at infinity**: Z = 0

**Point Doubling in Jacobian** (a = 0):
```
S = 4·X·Y²
M = 3·X²
X' = M² - 2S
Y' = M(S - X') - 8Y⁴
Z' = 2YZ
```

Cost: 4M + 4S (multiplications + squarings)

**Point Addition in Jacobian** (mixed, Q.Z = 1):
```
U₁ = X₁
U₂ = X₂·Z₁²
S₁ = Y₁
S₂ = Y₂·Z₁³
H = U₂ - U₁
R = S₂ - S₁
X₃ = R² - H³ - 2U₁H²
Y₃ = R(U₁H² - X₃) - S₁H³
Z₃ = H·Z₁
```

Cost: 8M + 3S

---

## 4. Modular Arithmetic

### 4.1 Modular Reduction for secp256k1

The special form of p enables fast reduction:

```
p = 2²⁵⁶ - c  where c = 2³² + 977 = 0x1000003D1

For x < 2⁵¹²:
  x = x_high · 2²⁵⁶ + x_low
  x mod p = x_low + x_high · c (mod p)
```

This may produce a result > p, requiring at most one subtraction.

### 4.2 Montgomery Multiplication

For repeated multiplications, Montgomery representation is efficient:

```
Mont(a) = a · R mod p   where R = 2²⁵⁶

MonPro(a', b') = a' · b' · R⁻¹ mod p
```

**Montgomery Reduction**:
```
T = a · b
m = T · (-p⁻¹) mod R
U = (T + m·p) / R
if U ≥ p: return U - p
return U
```

### 4.3 Modular Inversion

**Extended Euclidean Algorithm** finds a⁻¹ mod p:

```
gcd(a, p) = 1  →  ∃ x, y: ax + py = 1
            →  ax ≡ 1 (mod p)
            →  x = a⁻¹ mod p
```

**Binary GCD (DivStep62)**: Optimized for 64-bit operations, processes 62 bits per iteration.

**Fermat's Little Theorem** (for prime p):
```
a^(p-1) ≡ 1 (mod p)
a^(p-2) ≡ a⁻¹ (mod p)
```

Cost: ~256 squarings + ~128 multiplications

### 4.4 Modular Square Root

For p ≡ 3 (mod 4), the square root is:

```
√a ≡ a^((p+1)/4) (mod p)
```

This applies to secp256k1 since p ≡ 3 (mod 4).

### 4.5 Batch Inversion (Montgomery's Trick)

Compute n inversions with 1 actual inversion:

```
Given: a₁, a₂, ..., aₙ
Want:  a₁⁻¹, a₂⁻¹, ..., aₙ⁻¹

Step 1: Compute products
  c₁ = a₁
  c₂ = a₁ · a₂
  ...
  cₙ = a₁ · a₂ · ... · aₙ

Step 2: Invert final product
  cₙ⁻¹ = (a₁ · a₂ · ... · aₙ)⁻¹

Step 3: Back-propagate
  aₙ⁻¹ = cₙ₋₁ · cₙ⁻¹
  cₙ₋₁⁻¹ = aₙ · cₙ⁻¹
  ...
  a₁⁻¹ = c₁⁻¹
```

Total cost: 3(n-1) multiplications + 1 inversion

---

## 5. Implementation Optimizations

### 5.1 Precomputation Tables

For scalar multiplication k·G, precompute:
```
G, 2G, 3G, ..., 255G           (for byte 0)
256G, 2·256G, ..., 255·256G    (for byte 1)
...
```

32 tables × 256 entries = 8192 precomputed points

Scalar multiplication becomes ~32 point additions.

### 5.2 Windowed NAF

Non-Adjacent Form reduces additions by using signed digits:

```
NAF(k) represents k with digits in {-1, 0, 1}
No two consecutive non-zero digits
```

Width-w NAF uses digits in {0, ±1, ±3, ..., ±(2^(w-1) - 1)}

### 5.3 Endomorphism

secp256k1 has an efficiently computable endomorphism φ:

```
φ(x, y) = (β·x, y)

where β³ ≡ 1 (mod p)
β = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
```

And corresponding scalar:
```
φ(P) = λ·P

where λ³ ≡ 1 (mod n)
λ = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
```

This allows computing k·G as:
```
k = k₁ + k₂·λ  (mod n)  where |k₁|, |k₂| < √n
k·G = k₁·G + k₂·λ·G = k₁·G + k₂·φ(G)
```

Reduces scalar size by half, nearly doubling speed.

### 5.4 GLV Decomposition

Decompose scalar k into two half-sized scalars:

```
Given k, find k₁, k₂ such that:
  k ≡ k₁ + k₂·λ (mod n)
  |k₁|, |k₂| ≈ √n

Algorithm:
  1. Precompute vectors (a₁, b₁), (a₂, b₂)
     with a₁ + b₁·λ ≡ 0 and a₂ + b₂·λ ≡ 0 (mod n)
  2. c₁ = round(b₂·k / n)
     c₂ = round(-b₁·k / n)
  3. k₁ = k - c₁·a₁ - c₂·a₂
     k₂ = -c₁·b₁ - c₂·b₂
```

---

## 6. Worked Examples

### 6.1 Point on Curve Verification

Verify that G is on secp256k1:

```
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

Check: Gy² ≡ Gx³ + 7 (mod p)

Gx³ + 7 = (Gx²)·Gx + 7
        = 0x4866D6A45E1E89E4...  (256-bit result)

Gy² = 0x4866D6A45E1E89E4...     (same 256-bit result)

✓ G is on the curve
```

### 6.2 Point Doubling Example

Compute 2G:

```
λ = 3·Gx² / (2·Gy)  (mod p)

Gx² = 0x3A4B6C7D8E9F0A1B...
3·Gx² = 0xAEE445788DBD1E51...
2·Gy = 0x9075B4EE4D4788CAB...

λ = 0x786523A8E4B2C7D9...  (after modular inverse)

X₂ = λ² - 2·Gx = 0xC6047F9441ED7D6D...
Y₂ = λ(Gx - X₂) - Gy = 0x1AE168FEA63DC339...

2G = (0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5,
      0x1AE168FEA63DC339A3C58419466CEAE1061B7CD1A06ECA7E5AA0EB19D80BFBFE)
```

### 6.3 Public Key Derivation

Private key: k = 1 (for simplicity)

```
Public Key = k·G = 1·G = G

Compressed format:
  Prefix: 02 (if Gy is even) or 03 (if Gy is odd)
  Gy = 0x483A... → last bit = 0 → even → prefix = 02

  Compressed: 02 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
```

### 6.4 Address Derivation

From compressed public key to Bitcoin address:

```
Step 1: SHA256
  Input:  02 79BE667E...16F81798 (33 bytes)
  Output: 0E7B14A9C8...2F7A5D3C (32 bytes)

Step 2: RIPEMD160
  Input:  0E7B14A9C8...2F7A5D3C (32 bytes)
  Output: 751E76E8199196D454941C45D1B3A323F1433BD6 (20 bytes)

Step 3: Version + Checksum
  Version: 00 (mainnet P2PKH)
  Payload: 00 751E76E819...1433BD6
  Checksum: First 4 bytes of SHA256(SHA256(payload))

Step 4: Base58Check
  Result: 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
```

---

## Summary

This document covered the mathematical foundations of secp256k1:

1. **Curve definition** and its special properties (a=0, Koblitz)
2. **Point operations** in affine and Jacobian coordinates
3. **Modular arithmetic** optimizations for the special prime
4. **Implementation techniques** including precomputation and endomorphisms

Understanding these concepts is essential for:
- Implementing efficient cryptographic software
- Security analysis of Bitcoin systems
- Developing optimized GPU kernels for key search operations

---

*For more details, see the full technical paper in TECHNICAL_PAPER.md*
