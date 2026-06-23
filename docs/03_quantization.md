# Module 2: Quantization — FP32 to Fixed-Point

## 1. Why Quantize?

FPGA DSP slices operate on **integer** data. Floating-point requires significant additional logic.

| Precision | DSP Slices | LUTs | Power | Performance |
|-----------|-----------|------|-------|-------------|
| FP32 | ~4 per MAC | ~200 | High | 1× |
| Q8.8 (16-bit) | 1 per MAC | ~0 | Low | ~4× |
| Binary (1-bit) | 0 | ~10 | Ultra-low | ~10× |

## 2. Q-Format Notation

**Qm.n** = Signed fixed-point with `m` integer bits and `n` fractional bits.

Total bits = 1 (sign) + m + n

### Q8.8 Format

```
  Bit:  15  14  13  12  11  10   9   8 |  7   6   5   4   3   2   1   0
       ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
       │ S │ 2⁶│ 2⁵│ 2⁴│ 2³│ 2²│ 2¹│ 2⁰│2⁻¹│2⁻²│2⁻³│2⁻⁴│2⁻⁵│2⁻⁶│2⁻⁷│2⁻⁸│
       └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        Sign    Integer part (8 bits)      Fraction part (8 bits)

Range:     -128.0 to +127.99609375
Precision: 0.00390625 (2⁻⁸)
```

### Conversion

```
FP32 → Q8.8:   q = round(fp * 256)
Q8.8 → FP32:   fp = q / 256

Examples:
   1.5   → round(1.5 × 256) = 384  = 0x0180
   0.25  → round(0.25 × 256) = 64  = 0x0040
  -1.0   → round(-1.0 × 256) = -256 = 0xFF00
   3.141 → round(3.141 × 256) = 804 = 0x0324 (π approximation)
```

## 3. MAC in Q8.8

```
weight:    Q8.8  (16-bit signed)
activation: Q8.8  (16-bit signed)

product = weight × activation
        = Q8.8 × Q8.8
        = Q16.16  (32-bit signed)

accumulator = Σ(product)
            = Q16.16

result = accumulator[23:8]
       = Q16.16 → Q8.8 (truncation)

Saturation check:
  if accumulator[31:23] == all 0s or all 1s → no overflow
  else → saturate to 16'h7FFF or 16'h8000
```

### Bit Width After Each Operation

```
Input (Q8.8):    s[15:0]   ×   s[15:0]   =   s[31:0] (Q16.16)
                  weight        activation      product

Product:         s[31:0]   +   s[31:0]   =   s[31:0] (Q16.16)
                  product       accumulator     accumulator

Output:          s[31:0]  →  s[15:0] (Q8.8)
                 acc              result
                 (select bits [23:8] with saturation)
```

## 4. Weight Distribution Analysis

Typical MLP weight distribution:

```
Distribution of FC1 weights (FP32):
  Mean:   0.0023
  Std:    0.085
  Min:   -0.45
  Max:    0.42
  Range fits well within Q8.8 [-128.0, 127.996]
```

**Observation**: Neural network weights are typically small (< |1.0|), so Q8.8 provides ample integer range with good fractional precision.

## 5. Quantization Error

```
SNR = 10·log₁₀(σ²/σₑ²)

Where:
  σ²  = signal power (FP32)
  σₑ² = quantization error power

For Q8.8:
  Theoretical SNR ≈ 6.02 × 8 + 1.76 = 49.92 dB
  (6.02n + 1.76 dB for n fractional bits)
```

## 6. Weight Export Flow

```
PyTorch (.pth)
    │
    ▼
FP32 Weights (NumPy .npz)
    │
    ▼  × 256, round, clip
Q8.8 Weights (int16)
    │
    ├──→ .coe files (Xilinx BRAM init)
    ├──→ .hex files (Verilog $readmemh)
    └──→ .vh files (Verilog header for biases)
```

## 7. Accuracy Comparison

| Test | FP32 | Q8.8 | Diff |
|------|------|------|------|
| MNIST Test Set | 97.2% | 96.8% | -0.4% |
| Single Image 0 | 100% | 100% | 0% |
| Single Image 4 | 100% | 100% | 0% |
| Single Image 8 | 82% | 82% | 0% |

**Conclusion**: Q8.8 quantization introduces negligible accuracy loss.
