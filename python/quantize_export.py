"""
quantize_export.py
==================
Quantize trained MLP weights (FP32 → Q8.8 fixed-point) and generate:
  - .coe files for Xilinx BRAM initialization
  - .vh (Verilog header) files for bias parameters
  - .hex test image files for simulation

Q8.8 Format:
  [Signed] [8-bit Integer] . [8-bit Fraction]
  Range: -128.0 to +127.99609375
  Precision: 0.00390625 (2^-8)

Usage:
    python quantize_export.py

Requires: trained weights from train_mnist_mlp.py
"""

import numpy as np
import os
import struct


# =========================================================================
#  Configuration
# =========================================================================
class Config:
    # Network dimensions (must match train_mnist_mlp.py)
    input_size = 784
    hidden1_size = 64
    hidden2_size = 32
    num_classes = 10

    # Quantization format
    Q_FRACTION_BITS = 8
    Q_TOTAL_BITS = 16

    # Paths
    weight_file = "models/mnist_mlp_weights.npz"
    coe_dir = "../coe"
    num_test_images = 5  # number of test images to export


# =========================================================================
#  Q8.8 Quantization Functions
# =========================================================================
def quantize_q88(tensor_fp32, fraction_bits=Config.Q_FRACTION_BITS,
                 total_bits=Config.Q_TOTAL_BITS):
    """Convert FP32 tensor to Qm.n fixed-point integer.

    Args:
        tensor_fp32:     FP32 NumPy array
        fraction_bits:   Number of fractional bits (n)
        total_bits:      Total bit width (m+n+1 for signed)

    Returns:
        int16 NumPy array in Qm.n format
    """
    scale = 1 << fraction_bits  # 2^n
    max_val = (1 << (total_bits - 1)) - 1   # 32767
    min_val = -(1 << (total_bits - 1))      # -32768

    tensor_q = np.round(tensor_fp32 * scale)
    tensor_q = np.clip(tensor_q, min_val, max_val)
    return tensor_q.astype(np.int16)


def dequantize_q88(tensor_q, fraction_bits=Config.Q_FRACTION_BITS):
    """Convert Qm.n fixed-point back to FP32 for verification."""
    return tensor_q.astype(np.float32) / (1 << fraction_bits)


def quantization_error(original, quantized_q, fraction_bits=Config.Q_FRACTION_BITS):
    """Compute SNR and max error between FP32 and quantized values."""
    reconstructed = dequantize_q88(quantized_q, fraction_bits)
    error = original - reconstructed
    mse = np.mean(error ** 2)
    signal_power = np.mean(original ** 2)
    snr = 10 * np.log10(signal_power / (mse + 1e-30))
    max_err = np.max(np.abs(error))
    return snr, max_err, reconstructed


# =========================================================================
#  COE File Generation (for Xilinx BRAM)
# =========================================================================
def generate_coe(weights_q, layer_name, output_dir):
    """Generate a Xilinx COE file for BRAM initialization.

    COE format:
        memory_initialization_radix=16;
        memory_initialization_vector=
        value1
        value2
        ...
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{layer_name}.coe")

    flat = weights_q.flatten()
    with open(filepath, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for i, val in enumerate(flat):
            # Convert signed int16 to unsigned hex (2's complement)
            hex_val = val & 0xFFFF
            if i < len(flat) - 1:
                f.write(f"{hex_val:04X}\n")
            else:
                f.write(f"{hex_val:04X};\n")

    print(f"  [COE] {filepath}  ({flat.size} values, "
          f"{flat.size * 2 / 1024:.1f} KB)")
    return filepath


def generate_hex(weights_q, layer_name, output_dir):
    """Generate a hex memory file for Verilog $readmemh.

    One 16-bit hex value per line.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{layer_name}.hex")

    flat = weights_q.flatten()
    with open(filepath, "w") as f:
        for val in flat:
            f.write(f"{(val & 0xFFFF):04X}\n")

    print(f"  [HEX] {filepath}  ({flat.size} values)")
    return filepath


# =========================================================================
#  Verilog Header File Generation (for bias parameters)
# =========================================================================
def generate_vh_header(bias_q, layer_name, param_name, output_dir):
    """Generate a Verilog header file with bias as localparam array.

    Usage in Verilog:
        `include "fc1_bias.vh"
        always @(*) bias = fc1_bias[neuron_idx];
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{layer_name}_bias.vh")

    with open(filepath, "w") as f:
        f.write(f"// {layer_name}_bias.vh — automatically generated\n")
        f.write(f"// Q8.8 signed fixed-point bias values\n")
        f.write(f"localparam integer {param_name}_SIZE = {bias_q.size};\n\n")
        f.write(f"localparam signed [15:0] {param_name} [{bias_q.size}] = '{{\n")
        for i, val in enumerate(bias_q):
            hex_val = val & 0xFFFF
            comma = "," if i < bias_q.size - 1 else ""
            f.write(f"    16'h{hex_val:04X}{comma}  // bias[{i}] = {val / 256.0:.6f}\n")
        f.write("};\n")

    print(f"  [VH]  {filepath}  ({bias_q.size} biases)")
    return filepath


# =========================================================================
#  Test Image Export (for simulation)
# =========================================================================
def export_test_images(test_dataset, index_list, output_dir):
    """Export MNIST test images as Q8.8 hex files for Verilog testbench."""
    from torchvision import transforms

    os.makedirs(output_dir, exist_ok=True)

    for idx in index_list:
        image, label = test_dataset[idx]
        image_np = image.squeeze().numpy()  # (28, 28), range [0,1]

        # Flatten to 784 pixels
        flat = image_np.flatten()

        # Scale to Q8.8: input range [0, 1] → Q8.8
        # MNIST pixels are normalized: mean=0.1307, std=0.3081
        # We store raw pixel value * 256 as Q8.8
        # For simplicity, use (pixel * 256) rounded
        pixel_q88 = np.round(flat * 256).clip(0, 32767).astype(np.int16)

        # Write hex file
        hex_path = os.path.join(output_dir, f"test_image_{idx}.hex")
        with open(hex_path, "w") as f:
            for px in pixel_q88:
                f.write(f"{(px & 0xFFFF):04X}\n")

        # Write label file
        label_path = os.path.join(output_dir, f"test_label_{idx}.txt")
        with open(label_path, "w") as f:
            f.write(f"{label}\n")

        print(f"  [IMG] {hex_path}  (label={label})")

    # Create a combined image file for all test images
    write_combined_hex(test_dataset, index_list, output_dir)


def write_combined_hex(test_dataset, indices, output_dir):
    """Write a combined hex file (one image after another) for testbench."""
    combined_path = os.path.join(output_dir, "test_images_batch.hex")
    labels_path = os.path.join(output_dir, "test_labels.hex")

    with open(combined_path, "w") as f_img, open(labels_path, "w") as f_lbl:
        for idx in indices:
            image, label = test_dataset[idx]
            image_np = image.squeeze().numpy().flatten()
            pixel_q88 = np.round(image_np * 256).clip(0, 32767).astype(np.int16)

            for px in pixel_q88:
                f_img.write(f"{(px & 0xFFFF):04X}\n")
            f_lbl.write(f"{label:04X}\n")

    print(f"  [IMG] {combined_path}  ({len(indices)} images)")
    print(f"  [LBL] {labels_path}")


# =========================================================================
#  Golden Reference Generation (for testbench comparison)
# =========================================================================
def generate_golden_reference(weights, biases, test_dataset, indices, output_dir):
    """Run inference in FP32 and Q8.8, save golden results for testbench."""
    w1, b1, w2, b2, w3, b3 = weights["w1"], weights["b1"], \
                               weights["w2"], weights["b2"], \
                               weights["w3"], weights["b3"]

    # Quantize weights
    w1_q = quantize_q88(w1)
    b1_q = quantize_q88(b1)
    w2_q = quantize_q88(w2)
    b2_q = quantize_q88(b2)
    w3_q = quantize_q88(w3)
    b3_q = quantize_q88(b3)

    golden_path = os.path.join(output_dir, "golden_results.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(golden_path, "w") as f:
        f.write("# idx  expected  fp32_pred  q88_pred  match\n")

        for img_idx in indices:
            image, expected_label = test_dataset[img_idx]
            pixel = image.squeeze().numpy().flatten()

            # FP32 inference
            fp32_pred = mlp_forward_fp32(pixel, w1, b1, w2, b2, w3, b3)
            fp32_label = np.argmax(fp32_pred)

            # Q8.8 inference
            pixel_q88 = np.round(pixel * 256).clip(0, 32767).astype(np.int16)
            q88_pred = mlp_forward_q88(pixel_q88, w1_q, b1_q, w2_q, b2_q, w3_q, b3_q)
            q88_label = np.argmax(q88_pred)

            match = "PASS" if q88_label == expected_label else "FAIL"
            f.write(f"{img_idx}  {expected_label}  {fp32_label}  {q88_label}  {match}\n")

            print(f"  [GOLDEN] img {img_idx}: expected={expected_label}, "
                  f"fp32={fp32_label}, q88={q88_label}  {match}")

    print(f"  [GOLD] {golden_path}")


def mlp_forward_fp32(x, w1, b1, w2, b2, w3, b3):
    """FP32 MLP forward pass (reference)."""
    def relu(z):
        return np.maximum(0, z)

    h1 = relu(np.dot(w1, x) + b1)   # 64 = (64,784)·(784,) + (64,)
    h2 = relu(np.dot(w2, h1) + b2)  # 32 = (32,64)·(64,) + (32,)
    out = np.dot(w3, h2) + b3        # 10 = (10,32)·(32,) + (10,)
    return out


def mlp_forward_q88(x_q, w1_q, b1_q, w2_q, b2_q, w3_q, b3_q):
    """Q8.8 MLP forward pass (bit-accurate hardware model)."""
    def relu_q(z):
        return np.where(z < 0, 0, z)

    def mac_q(w, a):
        """Q8.8 MAC: sum(w[i] * a[i]) with saturation."""
        product = w.astype(np.int32) * a.astype(np.int32)  # 16×16 → 32-bit
        acc = np.sum(product, axis=1)                       # accumulate
        # Rescale from Q16.16 to Q8.8
        acc_q88 = np.round(acc / 256).astype(np.int32)
        # Saturation
        acc_q88 = np.clip(acc_q88, -32768, 32767)
        return acc_q88.astype(np.int16)

    h1 = relu_q(mac_q(w1_q, x_q) + b1_q)   # 64 neurons
    h2 = relu_q(mac_q(w2_q, h1) + b2_q)    # 32 neurons
    out = mac_q(w3_q, h2) + b3_q             # 10 classes
    return out


# =========================================================================
#  Quantization Analysis Report
# =========================================================================
def print_analysis(weights, weights_q):
    """Print quantization error analysis."""
    print("\n  Quantization Analysis:")
    print(f"  {'Layer':<10} {'MSE':<15} {'SNR (dB)':<12} {'Max Error':<12}")
    print(f"  {'-'*50}")

    layers = [
        ("FC1_W", weights["w1"], weights_q[0]),
        ("FC1_B", weights["b1"], weights_q[1]),
        ("FC2_W", weights["w2"], weights_q[2]),
        ("FC2_B", weights["b2"], weights_q[3]),
        ("FC3_W", weights["w3"], weights_q[4]),
        ("FC3_B", weights["b3"], weights_q[5]),
    ]

    for name, orig, quant in layers:
        snr, max_err, recon = quantization_error(orig, quant)
        mse = np.mean((orig - recon) ** 2)
        print(f"  {name:<10} {mse:<15.8f} {snr:<12.2f} {max_err:<12.6f}")


# =========================================================================
#  Main
# =========================================================================
def main():
    from torchvision import datasets, transforms

    print("=" * 60)
    print("MLP Weight Quantization & Export Tool")
    print("=" * 60)

    # Load trained weights
    if not os.path.exists(Config.weight_file):
        print(f"[ERROR] Weights not found at {Config.weight_file}")
        print("        Run train_mnist_mlp.py first.")
        return

    weights = np.load(Config.weight_file)
    print(f"\n[INFO] Loaded: {Config.weight_file}")
    print(f"  w1: {weights['w1'].shape}  {weights['w1'].dtype}")
    print(f"  b1: {weights['b1'].shape}  {weights['b1'].dtype}")
    print(f"  w2: {weights['w2'].shape}  {weights['w2'].dtype}")
    print(f"  b2: {weights['b2'].shape}  {weights['b2'].dtype}")
    print(f"  w3: {weights['w3'].shape}  {weights['w3'].dtype}")
    print(f"  b3: {weights['b3'].shape}  {weights['b3'].dtype}")

    # Quantize all parameters
    print(f"\n[INFO] Quantizing to Q{16 - Config.Q_FRACTION_BITS}."
          f"{Config.Q_FRACTION_BITS}...")

    w1_q = quantize_q88(weights["w1"])
    b1_q = quantize_q88(weights["b1"])
    w2_q = quantize_q88(weights["w2"])
    b2_q = quantize_q88(weights["b2"])
    w3_q = quantize_q88(weights["w3"])
    b3_q = quantize_q88(weights["b3"])

    all_quantized = [w1_q, b1_q, w2_q, b2_q, w3_q, b3_q]

    # Print analysis
    print_analysis(weights, all_quantized)

    # Generate COE files (for BRAM)
    print(f"\n[INFO] Generating COE files...")
    for name, data in [("fc1_weight", w1_q), ("fc2_weight", w2_q),
                        ("fc3_weight", w3_q)]:
        generate_coe(data, name, Config.coe_dir)
        generate_hex(data, name, Config.coe_dir)

    # Generate VH files (bias as Verilog parameters)
    print(f"\n[INFO] Generating Verilog header files...")
    for name, data, param in [
        ("fc1", b1_q, "FC1_BIAS"),
        ("fc2", b2_q, "FC2_BIAS"),
        ("fc3", b3_q, "FC3_BIAS")
    ]:
        generate_vh_header(data, name, param, Config.coe_dir)

    # Export test images and golden reference
    print(f"\n[INFO] Exporting test images...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    test_indices = list(range(Config.num_test_images))
    export_test_images(test_dataset, test_indices, Config.coe_dir)

    print(f"\n[INFO] Generating golden reference...")
    generate_golden_reference(weights, all_quantized, test_dataset,
                               test_indices, Config.coe_dir)

    # Summary
    total_weights = w1_q.size + w2_q.size + w3_q.size
    total_biases = b1_q.size + b2_q.size + b3_q.size
    total_bits = (total_weights + total_biases) * Config.Q_TOTAL_BITS

    print(f"\n{'=' * 60}")
    print(f"Export Complete!")
    print(f"  Total weights: {total_weights:,}")
    print(f"  Total biases:  {total_biases:,}")
    print(f"  Memory (BRAM): {total_weights * 2 / 1024:.1f} KB")
    print(f"  Output dir:    {Config.coe_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
