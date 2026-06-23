# MLP Neural Network Accelerator on Zybo Zynq-7020

## FPGA-based MLP Inference Accelerator — Educational Curriculum

![Zybo Z7-20](https://digilent.com/media/zybo-z7-20-1.png)

> **Goal**: Learn the complete pipeline of **Training → Quantization → Verilog RTL Design → FPGA Implementation** of an MLP (Multi-Layer Perceptron) neural network on the Zybo Zynq-7020 board.

---

## 📋 Overview

### Target Audience
- Undergraduate/graduate students who have completed basic digital logic design
- Those with basic Verilog HDL knowledge
- Those interested in neural network fundamentals

### Prerequisites
- Verilog HDL (module, always, FSM, testbench)
- Basic digital logic (combinational/sequential circuits)
- Python basics (NumPy, PyTorch basics)
- Neural network basics (Perceptron, Forward Propagation)

### Learning Outcomes

Upon completing this course, students will be able to:

| Competency | Details |
|------------|---------|
| **MLP Understanding** | MLP structure, Forward Propagation mathematics |
| **Quantization** | FP32 → Fixed-point conversion, precision analysis |
| **Verilog RTL** | MAC, FSM, BRAM interface design |
| **Zynq Integration** | PS-PL integration, AXI DMA, ARM firmware |
| **Vivado Flow** | Synthesis/implementation/bitstream, ILA debugging |

---

## 🗂️ Repository Structure

```
MLP_Neural_Network/
├── README.md                       # This document — Full curriculum
├── docs/
│   ├── 01_setup_guide.md           # Vivado / Vitis installation & setup
│   ├── 02_mlp_theory.md            # MLP theory details
│   └── 03_quantization.md          # Quantization theory & lab
│
├── python/
│   ├── train_mnist_mlp.py          # PyTorch MLP training script
│   ├── quantize_export.py          # Quantization & COE/VH export
│   └── requirements.txt            # Python dependencies
│
├── verilog/
│   ├── mac_unit.v                  # MAC (Multiply-Accumulate) unit
│   ├── mlp_fsm_controller.v        # FSM-based MLP controller
│   ├── relu.v                      # ReLU activation function
│   ├── argmax.v                    # Argmax classifier
│   ├── bram_wrapper.v              # BRAM wrapper (weight storage)
│   ├── mlp_top.v                   # Top-level module
│   └── tb_mlp_top.v               # Integrated testbench
│
├── tcl/
│   ├── create_project.tcl          # Vivado project creation Tcl
│   └── block_design.tcl            # Block Design Tcl
│
├── vitis/
│   ├── main.c                      # ARM Cortex-A9 firmware
│   └── mlp_driver.h                # MLP accelerator driver
│
└── coe/
     └── (weight/bias files generated after training)
```

---

## 📚 Curriculum: 6 Modules (14–18 hours total)

---

### Module 0: Development Environment Setup (1 hour)

**Objective**: Install Xilinx Vivado and Vitis, prepare the Zybo board.

#### Topics
| Item | Description |
|------|-------------|
| 0.1 | Install Xilinx Vivado ML Standard (WebPACK) |
| 0.2 | Install Vitis Unified IDE |
| 0.3 | Install Zybo Z7-20 board files |
| 0.4 | Configure USB-JTAG drivers |
| 0.5 | First project: LED Blink synthesis & download |

#### Key Tcl Commands
```tcl
# Create an empty Vivado project
create_project -part xc7z020clg400-1 led_test ./led_test
set_property board_part digilentinc.com:zybo-z7-20:part0:1.0 [current_project]
create_fileset -srcset sources_1
create_fileset -constrset constrs_1
```

#### Checklist
- [ ] Vivado synthesis/implementation/bitstream successful
- [ ] Zybo board detected and programmed
- [ ] LED blinking verified

---

### Module 1: MLP Structure & Forward Propagation Theory (2 hours)

**Objective**: Understand the mathematical structure of MLP and its forward propagation.

#### 1.1 The Perceptron (Single Neuron)

```
Input:   x = [x₁, x₂, ..., xₙ]
Weights: w = [w₁, w₂, ..., wₙ]
Bias:    b

Output:  y = σ( Σ(wᵢ · xᵢ) + b )
```

- `Σ(wᵢ · xᵢ)`: Dot product = **MAC (Multiply-Accumulate)** operation
- `σ()`: Non-linear activation function (ReLU, Sigmoid, etc.)

#### 1.2 3-Layer MLP Architecture

```
Input Layer (784)            Output Layer (10)
    ● ◂── w₁ [64×784]         ●
    ●                          ●
    ● ──→ Hidden Layer 1 ──→  ●  ← argmax → Predicted Label
    ●    ●        (64)        ●
    ●    ● ◂── w₂ [32×64]     ●
    ●    ●                     ●
    ●    ● ──→ Hidden Layer 2  ●
    ●    ●    ● ◂── w₃ [10×32]
    └ReLU┘  └ReLU┘       └(Identity)┘
```

**Layer MAC Count**:
| Layer | MAC Count | Weight Size |
|-------|-----------|-------------|
| FC1: 784 → 64 | 784 × 64 = **50,176** | 784×64 × 2B = 100,352 B |
| FC2: 64 → 32 | 64 × 32 = **2,048** | 64×32 × 2B = 4,096 B |
| FC3: 32 → 10 | 32 × 10 = **320** | 32×10 × 2B = 640 B |
| **Total** | **52,544** | **~105 KB** |

#### 1.3 Forward Propagation Equations

```
z¹ = W¹ · x  + b¹    (FC1: linear transform)
a¹ = ReLU(z¹)        (activation function)

z² = W² · a¹ + b²    (FC2)
a² = ReLU(z²)

z³ = W³ · a² + b³    (FC3 — output layer)
ŷ = softmax(z³)       (probability — only argmax used in FPGA)
```

#### Lab Exercise
- Implement MLP manually in Python (NumPy only)
- Run inference on MNIST image using pre-trained weights
- Save result as "Golden Reference" for hardware verification

```python
import numpy as np

def forward(x, W, b, activation='relu'):
    z = np.dot(W, x) + b
    if activation == 'relu':
        return np.maximum(0, z)
    return z

# 3-layer MLP forward pass
h1 = forward(x, W1, b1, 'relu')     # 784→64
h2 = forward(h1, W2, b2, 'relu')    # 64→32
out = forward(h2, W3, b3, 'none')   # 32→10
pred = np.argmax(out)
```

---

### Module 2: MNIST Training & Quantization (2 hours)

**Objective**: Train an MLP with PyTorch and quantize for FPGA deployment.

#### 2.1 PyTorch MLP Training

`python/train_mnist_mlp.py`:
- MNIST dataset (28×28 → 784 flattened)
- 3-layer MLP: 784 → 64 → 32 → 10
- CrossEntropyLoss + Adam Optimizer
- 10 epochs training → ~97% test accuracy

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # CrossEntropyLoss includes softmax
```

#### 2.2 Fixed-Point Quantization (Q8.8)

**Why Quantize?**

| Aspect | FP32 (Software) | Q8.8 (Hardware) |
|--------|----------------|-----------------|
| Bit width | 32-bit | 16-bit |
| DSP usage | DSP48 + many LUTs | Single DSP48 |
| BRAM efficiency | 4× capacity needed | 2× capacity |
| Speed | Slow (FPU required) | Fast (integer only) |

**Q8.8 Fixed-Point Format**:
```
[Signed] [8-bit Integer] . [8-bit Fraction]

Example:  1.5 = 0x0180 = 384 (Q8.8)
         0.25 = 0x0040 = 64
        -1.0 = 0xFF00 = -256
```

**Quantization Procedure**:

```python
# python/quantize_export.py

def quantize_q88(tensor_fp32):
    """FP32 tensor → Q8.8 fixed-point conversion"""
    scale = 2**8  # 256
    tensor_q = (tensor_fp32 * scale).round()
    tensor_q = np.clip(tensor_q, -32768, 32767)
    return tensor_q.astype(np.int16)

def generate_coe(weights_q, layer_name):
    """Q8.8 weights → Xilinx BRAM COE file"""
    coe_lines = ["radix=16;", "memory_initialization_radix=16;",
                  "memory_initialization_vector="]
    for w in weights_q.flatten():
        coe_lines.append(f"{w & 0xFFFF:04X}")
    return "\n".join(coe_lines)
```

#### Lab Exercise
1. Run `train_mnist_mlp.py` to save trained weights
2. Run `quantize_export.py` to generate COE/VH files
3. Compare FP32 vs Q8.8 inference accuracy

#### Output Files
| File | Contents |
|------|----------|
| `coe/fc1_weight.coe` | FC1 weights (784×64 × 16-bit) |
| `coe/fc2_weight.coe` | FC2 weights (64×32 × 16-bit) |
| `coe/fc3_weight.coe` | FC3 weights (32×10 × 16-bit) |
| `coe/fc1_bias.vh` | FC1 bias (Verilog header) |
| `coe/fc2_bias.vh` | FC2 bias |
| `coe/fc3_bias.vh` | FC3 bias |
| `coe/image_hex.txt` | Test images (hex format) |

---

### Module 3: Verilog RTL Design — Core Modules (4 hours)

**Objective**: Design the core MLP inference accelerator modules in Verilog.

#### 3.1 MAC Unit (`verilog/mac_unit.v`)

The MAC is the computational heart of the MLP. It performs multiplication and accumulation in a single clock cycle.

```verilog
// mac_unit.v — Single MAC (Multiply-Accumulate)
// Q8.8 Signed Fixed-Point operation
// Uses (* use_dsp = "yes" *) attribute for DSP48E1 inference

module mac_unit #(
    parameter DATA_WIDTH = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,
    input  wire                     clear_acc,
    input  wire signed [DATA_WIDTH-1:0] weight,
    input  wire signed [DATA_WIDTH-1:0] activation,
    output reg  signed [DATA_WIDTH-1:0] result,
    output wire                     overflow
);

    (* use_dsp = "yes" *)
    reg signed [DATA_WIDTH*2-1:0] accumulator;

    wire signed [DATA_WIDTH*2-1:0] product;
    assign product = weight * activation;  // 16×16 → 32-bit

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            result      <= 0;
        end else if (clear_acc) begin
            accumulator <= 0;
        end else if (enable) begin
            accumulator <= accumulator + product;
        end
    end

    // Q8.8 → Q8.8 with saturation
    always @(*) begin
        if (accumulator[31:15] != {accumulator[31], {16{1'b0}}})
            result <= accumulator[31] ? 16'h8000 : 16'h7FFF;
        else
            result <= accumulator[23:8];
    end

    assign overflow = (accumulator[31:15] != {accumulator[31], {16{1'b0}}});

endmodule
```

**Key Learning Points**:
- `(* use_dsp = "yes" *)` → maps to DSP48E1 slice
- 16×16 multiply → 32-bit product → upper 16-bit (Q8.8 × Q8.8 = Q16.16)
- Saturation logic prevents overflow

#### 3.2 FSM Controller (`verilog/mlp_fsm_controller.v`)

The FSM is the **central controller** for the entire MLP inference process.

**State Transition Diagram**:

```
         ┌──────────┐
         │   IDLE   │ ←─ waits for start signal
         └────┬─────┘
              │ start=1
         ┌────▼─────┐
         │ LOAD_IN  │ ←─ load input image (784 pixels)
         └────┬─────┘
              │
         ┌────▼─────┐      ┌──────────────┐
         │ MAC_FC1  │ ←──→ │  50,176 MACs  │
         └────┬─────┘      └──────────────┘
              │ FC1 done
         ┌────▼─────┐
         │ BIAS+REL1│ ←── add bias + apply ReLU
         └────┬─────┘
              │
         ┌────▼─────┐      ┌──────────────┐
         │ MAC_FC2  │ ←──→ │   2,048 MACs  │
         └────┬─────┘      └──────────────┘
              │ FC2 done
         ┌────▼─────┐
         │ BIAS+REL2│ ←── bias + ReLU
         └────┬─────┘
              │
         ┌────▼─────┐      ┌──────────────┐
         │ MAC_FC3  │ ←──→ │    320 MACs   │
         └────┬─────┘      └──────────────┘
              │ FC3 done
         ┌────▼─────┐
         │ BIAS_OUT │ ←── output bias (no activation)
         └────┬─────┘
              │
         ┌────▼─────┐
         │  ARGMAX  │ ←── find maximum index
         └────┬─────┘
              │
         ┌────▼─────┐
         │   DONE   │ ──→ done=1, result=predicted label
         └──────────┘
```

```verilog
// mlp_fsm_controller.v — MLP operation FSM controller

module mlp_fsm_controller #(
    parameter FC1_SIZE = 784 * 64,   // = 50176
    parameter FC2_SIZE = 64 * 32,    // = 2048
    parameter FC3_SIZE = 32 * 10     // = 320
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          done,
    output reg  [3:0]   state_debug
);

    // State encoding (one-hot recommended for Vivado)
    localparam IDLE      = 4'd0;
    localparam LOAD_IN   = 4'd1;
    localparam MAC_FC1   = 4'd2;
    localparam BIAS_REL1 = 4'd3;
    localparam MAC_FC2   = 4'd4;
    localparam BIAS_REL2 = 4'd5;
    localparam MAC_FC3   = 4'd6;
    localparam BIAS_OUT  = 4'd7;
    localparam ARGMAX    = 4'd8;
    localparam DONE      = 4'd9;

    reg [3:0] state, next_state;
    reg [31:0] mac_counter;
    wire mac_complete;

    // State Register (sequential)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    // Next State Logic (combinational)
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:      if (start)    next_state = LOAD_IN;
            LOAD_IN:                  next_state = MAC_FC1;
            MAC_FC1:  if (mac_complete) next_state = BIAS_REL1;
            BIAS_REL1:                next_state = MAC_FC2;
            MAC_FC2:  if (mac_complete) next_state = BIAS_REL2;
            BIAS_REL2:                next_state = MAC_FC3;
            MAC_FC3:  if (mac_complete) next_state = BIAS_OUT;
            BIAS_OUT:                 next_state = ARGMAX;
            ARGMAX:                   next_state = DONE;
            DONE:     if (!start)     next_state = IDLE;
        endcase
    end

    // MAC counter per layer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) mac_counter <= 0;
        else begin
            case (state)
                LOAD_IN:   mac_counter <=  0;
                MAC_FC1:   mac_counter <= (mac_counter < FC1_SIZE) ?
                                          mac_counter + 1 : mac_counter;
                MAC_FC2:   mac_counter <= (mac_counter < FC2_SIZE) ?
                                          mac_counter + 1 : mac_counter;
                MAC_FC3:   mac_counter <= (mac_counter < FC3_SIZE) ?
                                          mac_counter + 1 : mac_counter;
                default:   mac_counter <= 0;
            endcase
        end
    end

    assign mac_complete = (state == MAC_FC1 && mac_counter >= FC1_SIZE - 1) ||
                          (state == MAC_FC2 && mac_counter >= FC2_SIZE - 1) ||
                          (state == MAC_FC3 && mac_counter >= FC3_SIZE - 1);

    // Output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) done <= 0;
        else        done <= (state == DONE);
    end

    assign state_debug = state;

endmodule
```

**Key Learning Points**:
- **3-block FSM pattern**: State Register + Next State Logic + Output Logic
- **MAC counter** tracks operation progress
- Layer MAC counts are parameterized for reusability

#### 3.3 ReLU Activation (`verilog/relu.v`)

```verilog
// relu.v — ReLU: if (x < 0) return 0 else return x
// Simplest activation — just check the sign bit (MSB)

module relu #(
    parameter DATA_WIDTH = 16
) (
    input  wire signed [DATA_WIDTH-1:0] data_in,
    output reg  signed [DATA_WIDTH-1:0] data_out
);

    always @(*) begin
        if (data_in[DATA_WIDTH-1])   // MSB = 1 → negative
            data_out <= 0;
        else
            data_out <= data_in;
    end

endmodule
```

**Key Point**: ReLU is the **simplest hardware activation** — just check the MSB (sign bit). Zero DSP/BRAM overhead.

#### 3.4 Argmax (`verilog/argmax.v`)

```verilog
// argmax.v — Find index of maximum value among 10 output classes

module argmax #(
    parameter NUM_CLASSES = 10,
    parameter DATA_WIDTH  = 16
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,
    input  wire signed [DATA_WIDTH-1:0] scores [0:NUM_CLASSES-1],
    output reg  [3:0]                   predicted_class,
    output reg                          valid
);

    reg signed [DATA_WIDTH-1:0] max_val;
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            predicted_class <= 0;
            max_val         <= 0;
            valid           <= 0;
        end else if (enable) begin
            max_val <= scores[0];
            predicted_class <= 0;
            for (i = 1; i < NUM_CLASSES; i = i + 1) begin
                if (scores[i] > max_val) begin
                    max_val <= scores[i];
                    predicted_class <= i;
                end
            end
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end

endmodule
```

#### 3.5 BRAM Wrapper (`verilog/bram_wrapper.v`)

```verilog
// bram_wrapper.v — BRAM used as Weight ROM
// Compatible with Xilinx Block Memory Generator interface

module bram_wrapper #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16,
    parameter INIT_FILE  = "none"
) (
    input  wire                     clk,
    input  wire                     en,
    input  wire [ADDR_WIDTH-1:0]    addr,
    output reg  [DATA_WIDTH-1:0]    dout
);

    // Xilinx BRAM initialized via $readmemh
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    initial begin
        if (INIT_FILE != "none")
            $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        if (en)
            dout <= mem[addr];
    end

endmodule
```

#### 3.6 Top-Level Module (`verilog/mlp_top.v`) — System Integration

```verilog
// mlp_top.v — MLP Inference System Top-Level
// AXI4-Lite interface for ARM processor control

module mlp_top #(
    parameter IMG_SIZE     = 784,
    parameter HIDDEN1_SIZE = 64,
    parameter HIDDEN2_SIZE = 32,
    parameter NUM_CLASSES  = 10,
    parameter DATA_WIDTH   = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    // Control interface
    input  wire                     start,
    output wire                     done,
    output wire [3:0]               predicted,
    // Data interface
    input  wire [DATA_WIDTH-1:0]    input_data,
    input  wire                     input_valid,
    output wire                     input_ready
);

    // Internal signals
    wire mac_enable, mac_clr;
    wire [DATA_WIDTH-1:0] mac_result;
    wire [3:0] state;
    wire [DATA_WIDTH-1:0] current_weight, current_activation;

    // Predicted label output
    reg [3:0] predicted_reg;

    // --- FSM Controller ---
    mlp_fsm_controller #(
        .FC1_SIZE(IMG_SIZE * HIDDEN1_SIZE),
        .FC2_SIZE(HIDDEN1_SIZE * HIDDEN2_SIZE),
        .FC3_SIZE(HIDDEN2_SIZE * NUM_CLASSES)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done),
        .state_debug(state)
    );

    // --- BRAM for weights (instantiated per layer) ---
    bram_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH($clog2(IMG_SIZE * HIDDEN1_SIZE)),
        .INIT_FILE("../../coe/fc1_weight.hex")
    ) u_bram_fc1 (
        .clk(clk), .en(mac_enable),
        .addr(w_addr_fc1), .dout(current_weight)
    );

    // --- MAC Unit ---
    mac_unit #(.DATA_WIDTH(DATA_WIDTH)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .enable(mac_enable), .clear_acc(mac_clr),
        .weight(current_weight),
        .activation(current_activation),
        .result(mac_result), .overflow()
    );

    // --- ReLU (layer 1 & 2) ---
    relu #(.DATA_WIDTH(DATA_WIDTH)) u_relu (
        .data_in(mac_result_bias),
        .data_out(activation_out)
    );

    // --- Argmax ---
    argmax #(.NUM_CLASSES(NUM_CLASSES), .DATA_WIDTH(DATA_WIDTH)) u_argmax (
        .clk(clk), .rst_n(rst_n),
        .enable(state == ARGMAX),
        .scores(output_buffer),
        .predicted_class(predicted_class_wire),
        .valid()
    );

    // --- Address Generator (combinational) ---
    // Maps (layer, row, col) → BRAM address
    always @(*) begin
        case (state)
            MAC_FC1: begin
                w_addr_fc1 = mac_counter;
                // ... mux activation from input buffer
            end
            MAC_FC2: begin
                w_addr_fc2 = mac_counter;
                // ... mux activation from layer1 buffer
            end
            // ...
        endcase
    end

    // Output register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)    predicted_reg <= 0;
        else if (done) predicted_reg <= predicted_class_wire;
    end

    assign predicted = predicted_reg;

endmodule
```

---

### Module 4: Zynq PS-PL Integration (2 hours)

**Objective**: Integrate the ARM processor (PS) with the FPGA logic (PL) to build a complete system.

#### 4.1 Zynq-7020 Architecture

```
┌─────────────────────────────────────────────────┐
│ Zynq-7020 (XC7Z020)                              │
│                                                   │
│ ┌─── Processing System (PS) ──────────────────┐  │
│ │  ARM Cortex-A9 #0     ARM Cortex-A9 #1      │  │
│ │        @ 667 MHz          @ 667 MHz          │  │
│ │                                              │  │
│ │  L1 Cache: 32KB I / 32KB D per core         │  │
│ │  L2 Cache: 512KB (shared)                   │  │
│ │  DDR3 Controller: 1GB @ 533MHz              │  │
│ │                                              │  │
│ │  ┌── AXI Interconnect ────────────────────┐  │  │
│ │  │  AXI_HP (4× 64-bit) — High-speed data   │  │  │
│ │  │  AXI_GP (2× 32-bit) — Control/status    │  │  │
│ │  │  AXI_ACP (1× 64-bit) — Cache coherent   │  │  │
│ │  └────────────────────────────────────────┘  │  │
│ └──────────────────────────────────────────────┘  │
│                        │ AXI interface             │
│ ┌─── Programmable Logic (PL) ─────────────────┐  │
│ │                                              │  │
│ │   MLP Accelerator Core (your Verilog design) │  │
│ │                                              │  │
│ │   BRAM (weights)   DSP48E1 (MAC)   FSM (ctrl)│  │
│ │                                              │  │
│ └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### 4.2 Vivado Block Design (`tcl/block_design.tcl`)

```tcl
# tcl/block_design.tcl — MLP System Block Design

# Create Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" \
    Master "Disable" Slave "Disable" } [get_bd_cells processing_system7_0]

# AXI GPIO (PS → PL control: start, input_data)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0
set_property -dict [list \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_DOUT_DEFAULT_0 {0x00000000} \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_ALL_OUTPUTS_1 {1} \
] [get_bd_cells axi_gpio_0]

# AXI GPIO (PL → PS status: done, predicted_class)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_1
set_property -dict [list \
    CONFIG.C_ALL_INPUTS {1} \
    CONFIG.C_GPIO_WIDTH {8} \
] [get_bd_cells axi_gpio_1]

# AXI SmartConnect (interconnect)
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] \
    [get_bd_cells smartconnect_0]

# MLP Accelerator (RTL module)
create_bd_cell -type module -reference mlp_top mlp_top_0

# Connect AXI bus
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { Clk_master {Auto} Clk_slave {Auto} \
    Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} \
    Slave {/smartconnect_0/S00_AXI} } [get_bd_intf_pins smartconnect_0/S00_AXI]

# Connect clocks and resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins mlp_top_0/clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins mlp_top_0/rst_n]

# Make external ports for debug
make_bd_pins_external [get_bd_pins mlp_top_0/predicted]
make_bd_pins_external [get_bd_pins mlp_top_0/done]

# Validate and generate
validate_bd_design
generate_target all [get_files  *.bd]
```

#### 4.3 ARM Firmware (`vitis/main.c`)

```c
// vitis/main.c — ARM Cortex-A9 Firmware
// Controls MLP accelerator and outputs results via UART

#include <stdio.h>
#include <xil_printf.h>
#include "xparameters.h"
#include "xgpio.h"
#include "sleep.h"

#define GPIO_CTRL_DEVICE_ID   XPAR_AXI_GPIO_0_DEVICE_ID
#define GPIO_STATUS_DEVICE_ID XPAR_AXI_GPIO_1_DEVICE_ID

XGpio gpio_ctrl, gpio_status;

// Test image data (generated by quantize_export.py)
// 784 Q8.8 fixed-point pixel values per image
#include "test_image.h"

int main() {
    int status, predicted, expected;
    u32 result;

    xil_printf("\r\n=== MLP Neural Network Accelerator ===\r\n");
    xil_printf("Zybo Zynq-7020 MNIST Inference Test\r\n");
    xil_printf("======================================\r\n\n");

    // Initialize GPIOs
    status = XGpio_Initialize(&gpio_ctrl, GPIO_CTRL_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("GPIO Ctrl init failed!\r\n");
        return XST_FAILURE;
    }

    status = XGpio_Initialize(&gpio_status, GPIO_STATUS_DEVICE_ID);
    if (status != XST_SUCCESS) {
        xil_printf("GPIO Status init failed!\r\n");
        return XST_FAILURE;
    }

    // Set GPIO direction
    XGpio_SetDataDirection(&gpio_ctrl, 1, 0x00000000);  // Channel 1: output
    XGpio_SetDataDirection(&gpio_ctrl, 2, 0x00000000);  // Channel 2: output
    XGpio_SetDataDirection(&gpio_status, 1, 0x000000FF); // Channel 1: input

    // Test loop
    int pass_count = 0;
    for (int img = 0; img < NUM_TEST_IMAGES; img++) {
        xil_printf("Test %2d/%d: ", img + 1, NUM_TEST_IMAGES);

        // Step 1: Write input data to PL (simplified — GPIO ch2)
        // In a full design, use AXI DMA for high-speed transfer
        for (int px = 0; px < 784; px++) {
            XGpio_DiscreteWrite(&gpio_ctrl, 2,
                                (u32)(test_images[img][px] & 0xFFFF));
            usleep(1);
        }

        // Step 2: Assert start signal
        XGpio_DiscreteWrite(&gpio_ctrl, 1, 0x00000001);  // start = 1
        usleep(10);
        XGpio_DiscreteWrite(&gpio_ctrl, 1, 0x00000000);  // start = 0

        // Step 3: Wait for done signal
        do {
            result = XGpio_DiscreteRead(&gpio_status, 1);
        } while (!(result & 0x01));  // bit 0 = done flag

        // Step 4: Read prediction (bits 4:7 = predicted class)
        predicted = (result >> 4) & 0x0F;
        expected  = test_labels[img];

        if (predicted == expected) {
            xil_printf("PASS  predicted=%d, expected=%d\r\n",
                       predicted, expected);
            pass_count++;
        } else {
            xil_printf("FAIL  predicted=%d, expected=%d\r\n",
                       predicted, expected);
        }
    }

    xil_printf("\n=== Results: %d/%d passed (%.1f%%) ===\r\n",
               pass_count, NUM_TEST_IMAGES,
               100.0 * pass_count / NUM_TEST_IMAGES);

    return 0;
}
```

#### 4.4 Driver Header (`vitis/mlp_driver.h`)

```c
// vitis/mlp_driver.h — MLP Accelerator Driver Header

#ifndef MLP_DRIVER_H
#define MLP_DRIVER_H

#include "xgpio.h"
#include "xparameters.h"

/* Register offsets (AXI GPIO bit assignments) */
#define MLP_CTRL_START_BIT     0  // bit 0: start inference
#define MLP_CTRL_RST_BIT       1  // bit 1: reset accelerator

#define MLP_STATUS_DONE_BIT    0  // bit 0: inference done
#define MLP_STATUS_CLASS_LSB   4  // bits 4-7: predicted class

/* MLP Accelerator Driver Functions */

void mlp_start(XGpio *gpio) {
    XGpio_DiscreteWrite(gpio, 1, 1 << MLP_CTRL_START_BIT);
    usleep(1);
    XGpio_DiscreteWrite(gpio, 1, 0);
}

int mlp_is_done(XGpio *gpio) {
    return (XGpio_DiscreteRead(gpio, 1) >> MLP_STATUS_DONE_BIT) & 1;
}

int mlp_get_prediction(XGpio *gpio) {
    return (XGpio_DiscreteRead(gpio, 1) >> MLP_STATUS_CLASS_LSB) & 0x0F;
}

void mlp_send_pixel(XGpio *gpio, short pixel_q88) {
    XGpio_DiscreteWrite(gpio, 2, (u32)(pixel_q88 & 0xFFFF));
}

int mlp_run_inference(XGpio *gpio_ctrl, XGpio *gpio_status,
                       short *image_q88) {
    // Send all 784 pixels
    for (int i = 0; i < 784; i++) {
        mlp_send_pixel(gpio_ctrl, image_q88[i]);
    }

    // Start inference
    mlp_start(gpio_ctrl);

    // Wait for completion
    while (!mlp_is_done(gpio_status));

    return mlp_get_prediction(gpio_status);
}

#endif /* MLP_DRIVER_H */
```

---

### Module 5: Simulation & Hardware Verification (2 hours)

**Objective**: Verify the MLP design through simulation and ILA hardware debugging.

#### 5.1 Testbench (`verilog/tb_mlp_top.v`)

```verilog
// tb_mlp_top.v — MLP integrated testbench
// Compares results against Python "Golden Reference"

`timescale 1ns / 1ps

module tb_mlp_top();

    reg         clk, rst_n, start;
    reg  [15:0] test_image [0:783];
    wire        done;
    wire [3:0]  predicted;

    // Clock: 100 MHz → 10 ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // DUT instantiation
    mlp_top #(
        .IMG_SIZE(784),
        .HIDDEN1_SIZE(64),
        .HIDDEN2_SIZE(32),
        .NUM_CLASSES(10)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .predicted(predicted)
    );

    // Test sequence
    initial begin
        integer i;
        integer errors, total;
        integer expected_label;
        reg [15:0] expected_q88 [0:9];
        integer fd;

        $display("========================================");
        $display("MLP Inference Testbench");
        $display("========================================");

        // Initialize
        rst_n = 0;
        start = 0;
        #100;
        rst_n = 1;
        #20;

        // Load test data
        $readmemh("../../coe/test_image_0.hex", test_image);

        // Set expected label for this image
        expected_label = 7;  // example only — read from golden file

        // Run inference
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Wait for completion
        wait(done);

        // Display result
        $display("TIME=%0t ns", $time);
        $display("Predicted: %0d  Expected: %0d  %s",
                 predicted, expected_label,
                 (predicted == expected_label) ? "PASS" : "FAIL");

        if (predicted == expected_label)
            $display("*** TEST PASSED ***");
        else
            $display("*** TEST FAILED ***");

        #100;
        $finish;
    end

endmodule
```

#### 5.2 ILA Hardware Debugging

```tcl
# Add ILA (Integrated Logic Analyzer) to the design
create_debug_core u_ila_0 ila
set_property -dict [list \
    CONFIG.C_NUM_OF_PROBES {4} \
    CONFIG.C_PROBE0_WIDTH {1} \
] [get_debug_cores u_ila_0]

# Connect signals to probes
connect_debug_port u_ila_0/probe0 [get_nets {start}]
connect_debug_port u_ila_0/probe1 [get_nets {done}]
connect_debug_port u_ila_0/probe2 [get_nets {state_debug}]
connect_debug_port u_ila_0/probe3 [get_nets {predicted}]

# Trigger configuration
set_property -dict [list CONFIG.C_TRIGIN_EN {true}] [get_debug_cores u_ila_0]
```

#### 5.3 Verification Flow

```
┌─────────────────────────────────────────────────────┐
│ Verification Stages                                   │
├─────────────────────────────────────────────────────┤
│                                                       │
│ Stage 1: Behavioral Simulation (RTL)                  │
│   - Vivado Simulator functional verification          │
│   - Compare against Python Golden Reference           │
│                                                       │
│ Stage 2: Post-Synthesis Simulation                    │
│   - Gate-level verification after synthesis           │
│   - Check timing constraints                          │
│                                                       │
│ Stage 3: Post-Implementation Simulation               │
│   - Post-routing simulation                           │
│   - Verify setup/hold timing                          │
│                                                       │
│ Stage 4: Hardware Debug (ILA)                         │
│   - Program FPGA with bitstream                       │
│   - Capture real-time signals with ILA                │
│   - Print final results via ARM UART                  │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

### Module 6: Advanced Topics — Performance Optimization (Optional, 2 hours)

**Objective**: Learn pipelining and parallelization techniques for performance improvement.

#### 6.1 Performance Analysis

**Serial MAC (current design)**:
- 1 cycle per MAC operation
- Total operations: 52,544 MAC
- Total time: 52,544 cycles @ 100 MHz ≈ **0.525 ms**

#### 6.2 Parallelization Strategies

| Approach | Latency | DSP Usage | Description |
|----------|---------|-----------|-------------|
| Serial (1 MAC) | 0.525 ms | 1 | Smallest, simplest |
| Semi-Parallel (P MACs) | 0.525/P ms | P | Resource/speed trade-off |
| Fully parallel (entire layer) | 64 cycles | 64+32+10=106 | Fastest, largest |

#### 6.3 Pipelined MAC Array

```verilog
// pipelined_mac_array.v — P-way parallel MAC array

module pipelined_mac_array #(
    parameter P = 8,       // parallelism factor
    parameter DW = 16
) (
    input  wire clk, rst_n,
    input  wire signed [DW-1:0] weights [0:P-1],
    input  wire signed [DW-1:0] activations [0:P-1],
    input  wire clear_acc,
    output reg  signed [DW-1:0] result
);

    genvar i;
    generate
        for (i = 0; i < P; i = i + 1) begin : mac_array
            mac_unit u_mac (
                .clk(clk), .rst_n(rst_n),
                .enable(1'b1),
                .clear_acc(clear_acc),
                .weight(weights[i]),
                .activation(activations[i]),
                .result()
            );
        end
    endgenerate

    // P MAC results are summed in a pipelined adder tree
    // ...

endmodule
```

#### 6.4 Performance Measurement

```c
// vitis/benchmark.c — Cycle-accurate performance measurement

#include "xttcps.h"

XTtcPs timer;

void measure_inference_time() {
    u64 t_start, t_end;

    XTtcPs_Start(&timer);
    t_start = XTtcPs_GetCounterValue(&timer);

    // Run inference
    start_inference();
    wait_for_done();

    t_end = XTtcPs_GetCounterValue(&timer);
    XTtcPs_Stop(&timer);

    u64 elapsed = t_end - t_start;
    double us = (double)elapsed / 100.0;  // 100 MHz → microseconds

    xil_printf("Inference: %.1f us (%llu cycles @ 100MHz)\r\n",
               us, elapsed);
}
```

---

## 🛠️ Lab Environment Setup

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Vivado ML Standard | 2023.1+ | Synthesis, implementation, bitstream |
| Vitis Unified IDE | 2023.1+ | ARM firmware development |
| Python | 3.8+ | MLP training and quantization |
| PyTorch | 2.0+ | MNIST MLP training |
| NumPy | 1.24+ | Quantization math |

### Required Hardware

| Item | Qty | Notes |
|------|-----|-------|
| Zybo Z7-20 board | 1 | Primary target |
| Micro-USB cable | 2 | Power + UART |
| microSD card (8GB+) | 1 | Boot image storage |
| HDMI monitor (optional) | 1 | Result visualization |
| Pcam camera (optional) | 1 | Real-time video input |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/MLP_Neural_Network.git
cd MLP_Neural_Network

# 2. Set up Python environment
pip install -r python/requirements.txt

# 3. Train and quantize MLP
cd python
python train_mnist_mlp.py
python quantize_export.py

# 4. Create Vivado project
vivado -mode tcl -source ../tcl/create_project.tcl

# 5. Build ARM firmware (in Vitis)
# Open vitis/ directory in Vitis IDE and build

# 6. Program FPGA and run
# - Load bitstream via Vivado Hardware Manager
# - Run ARM firmware via Vitis
```

---

## 📝 Grading Rubric

| Criteria | Weight | Passing Standard |
|----------|--------|-----------------|
| Python MLP training + quantization | 20% | Test accuracy ≥ 95% (Q8.8) |
| Verilog RTL simulation | 30% | 100% match with Golden Reference |
| Vivado synthesis success | 15% | LUT < 80%, BRAM < 90%, DSP < 90% |
| ARM firmware verification | 20% | Correct predictions via UART |
| ILA debug report | 15% | Waveform capture of key signals |

---

## 📖 Reference Materials

| Resource | Link |
|----------|------|
| Zybo Z7-20 Reference Manual | https://digilent.com/reference/programmable-logic/zybo-z7 |
| Vivado Design Suite User Guide | https://docs.xilinx.com/r/en-US/ug910-vivado-getting-started |
| Zynq-7000 TRM (UG585) | https://docs.xilinx.com/v/u/en-US/ug585-Zynq-7000-TRM |
| 7 Series DSP48E1 UG479 | https://docs.xilinx.com/v/u/en-US/ug479_7Series_DSP48E1 |
| MNIST Dataset | http://yann.lecun.com/exdb/mnist/ |
| PyTorch MLP Tutorial | https://pytorch.org/tutorials/beginner/basics/intro.html |

---

## 📄 License

This educational material is provided under the MIT License. Feel free to modify and distribute.

---

> **Author**: [Your Name]  
> **Contact**: [your.email@example.com]  
> **Last Updated**: 2026-06-23
