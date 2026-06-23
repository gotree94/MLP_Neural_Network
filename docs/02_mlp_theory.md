# Module 1: MLP Theory & Forward Propagation

## 1. The Perceptron

The perceptron is the fundamental building block of neural networks:

```
Input:  x = [x₁, x₂, ..., xₙ]
Weight: w = [w₁, w₂, ..., wₙ]
Bias:   b

Step 1: z = Σ(wᵢ · xᵢ) + b    ← dot product + bias
Step 2: y = σ(z)                ← activation function
```

### Matrix Form

For a layer with `m` neurons and `n` inputs:

```
z = W · x + b

Where:
  W: m×n weight matrix
  x: n×1 input vector
  b: m×1 bias vector
  z: m×1 pre-activation vector
  y: m×1 output (after activation)
```

## 2. Activation Functions

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

- **Hardware-friendly**: Only checks the sign bit (MSB)
- Derivative: 0 for z < 0, 1 for z > 0
- Used in hidden layers

### Softmax (output layer)

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

- Produces probability distribution (sums to 1.0)
- **Not implemented in FPGA**: Argmax is sufficient for classification
- Only needed for training loss computation

## 3. 3-Layer MLP Architecture

### Network: 784 → 64 → 32 → 10

```
Layer 1 (FC1):      z¹ = W¹ · x  + b¹    → 64 neurons
                     a¹ = ReLU(z¹)

Layer 2 (FC2):      z² = W² · a¹ + b²    → 32 neurons
                     a² = ReLU(z²)

Layer 3 (FC3):      z³ = W³ · a² + b³    → 10 logits
                     ŷ = argmax(z³)       → predicted digit (0-9)
```

### Parameter Count

| Layer | Weights | Biases | Total Parameters |
|-------|---------|--------|-----------------|
| FC1 (784→64) | 50,176 | 64 | 50,240 |
| FC2 (64→32) | 2,048 | 32 | 2,080 |
| FC3 (32→10) | 320 | 10 | 330 |
| **Total** | **52,544** | **106** | **52,650** |

### Memory Required (Q8.8)

- Weights: 52,544 × 2 bytes = **105,088 bytes ≈ 103 KB**
- Biases: 106 × 2 bytes = 212 bytes
- Zybo Z7-20 BRAM: 630 KB → **16% utilization** for weights

## 4. MAC Operations

Each neuron computes: **Σ(weight × activation) + bias**

This is a **Multiply-Accumulate (MAC)** operation.

| Layer | MACs per Inference | Percentage |
|-------|-------------------|-----------|
| FC1 | 50,176 | 95.5% |
| FC2 | 2,048 | 3.9% |
| FC3 | 320 | 0.6% |
| **Total** | **52,544** | **100%** |

FC1 dominates (95.5%) — optimization efforts should focus here.

## 5. MNIST Dataset

- 60,000 training images, 10,000 test images
- Each image: 28×28 = 784 grayscale pixels
- Pixel values: 0 (white) to 255 (black)
- Normalized: mean=0.1307, std=0.3081

## 6. Python Golden Reference

```python
import numpy as np

def mlp_forward(x, W1, b1, W2, b2, W3, b3):
    """FP32 reference implementation."""
    def relu(z): return np.maximum(0, z)

    h1 = relu(np.dot(W1, x) + b1)
    h2 = relu(np.dot(W2, h1) + b2)
    out = np.dot(W3, h2) + b3
    return out  # logits (not softmax)

def mlp_forward_q88(x_q, w1_q, b1_q, w2_q, b2_q, w3_q, b3_q):
    """Bit-accurate Q8.8 implementation (matches hardware)."""
    def relu_q(z): return np.where(z < 0, 0, z)

    def mac_q(w, a):
        prod = w.astype(np.int32) * a.astype(np.int32)
        acc = np.sum(prod, axis=1)
        acc_q88 = np.round(acc / 256).astype(np.int32)
        return np.clip(acc_q88, -32768, 32767).astype(np.int16)

    h1 = relu_q(mac_q(w1_q, x_q) + b1_q)
    h2 = relu_q(mac_q(w2_q, h1) + b2_q)
    out = mac_q(w3_q, h2) + b3_q
    return out
```

## 7. Expected Accuracy

| Precision | Expected Accuracy | Note |
|-----------|------------------|------|
| FP32 (software) | ~97% | Baseline |
| Q8.8 (hardware) | ~96-97% | <1% degradation |
| Q4.4 (low precision) | ~90-93% | Significant loss |
