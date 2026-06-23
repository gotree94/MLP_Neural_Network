//=============================================================================
// mac_unit.v — MAC (Multiply-Accumulate) Unit
//=============================================================================
//
// This is the computational core of the MLP accelerator.
// Performs: accumulator <= accumulator + (weight × activation)
//
// Fixed-point format: Q8.8 (signed 16-bit, 8 fractional bits)
//   - weight × activation = Q16.16 (32-bit)
//   - accumulator maintains Q16.16 precision
//   - output truncates to Q8.8 with saturation
//
// DSP48E1 Inference:
//   The (* use_dsp = "yes" *) synthesis attribute forces Vivado
//   to map this module to a DSP48E1 slice instead of LUT logic.
//
// Usage:
//   1. Assert clear_acc for 1 cycle to reset accumulator
//   2. Assert enable for each MAC operation
//   3. Read result after the desired number of MACs
//
//=============================================================================

module mac_unit #(
    parameter DATA_WIDTH = 16   // Q8.8 signed fixed-point
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,       // 1-cycle enable
    input  wire                         clear_acc,    // clear accumulator
    input  wire signed [DATA_WIDTH-1:0] weight,       // Q8.8 weight
    input  wire signed [DATA_WIDTH-1:0] activation,   // Q8.8 activation
    output reg  signed [DATA_WIDTH-1:0] result,       // Q8.8 output
    output wire                         overflow      // saturation flag
);

    //-------------------------------------------------------------------------
    // DSP48E1-style MAC (inferred via synthesis attribute)
    //-------------------------------------------------------------------------
    (* use_dsp = "yes" *)  // forces DSP48E1 mapping
    reg signed [DATA_WIDTH*2-1:0] accumulator;  // 32-bit: Q16.16

    wire signed [DATA_WIDTH*2-1:0] product;
    assign product = weight * activation;  // 16×16 signed → 32-bit

    // Accumulator
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= {DATA_WIDTH*2{1'b0}};
            result      <= {DATA_WIDTH{1'b0}};
        end else if (clear_acc) begin
            accumulator <= {DATA_WIDTH*2{1'b0}};
        end else if (enable) begin
            accumulator <= accumulator + product;
        end
    end

    //-------------------------------------------------------------------------
    // Output: Q16.16 → Q8.8 with saturation
    //-------------------------------------------------------------------------
    // The product is Q16.16 (weight × activation, both Q8.8).
    // We need to extract bits [23:8] to get back to Q8.8.
    // Saturation check: if accumulator[31:15] is not just sign extension,
    // the value overflowed and we saturate.
    wire signed [DATA_WIDTH-1:0] raw_result;
    assign raw_result = accumulator[23:8];  // truncate Q16.16 → Q8.8

    always @(*) begin
        // Check if accumulator exceeds Q8.8 range
        if (accumulator[31] == 1'b0) begin  // positive
            if (|accumulator[30:23])         // overflow (bits above Q8.8 are set)
                result <= 16'h7FFF;          // saturate to max positive
            else
                result <= raw_result;
        end else begin                      // negative
            if (&accumulator[30:23] == 1'b0) // overflow (bits above Q8.8 not all 1)
                result <= 16'h8000;          // saturate to max negative
            else
                result <= raw_result;
        end
    end

    assign overflow = (accumulator[31:23] != {accumulator[31], {7{accumulator[31]}}});

    //-------------------------------------------------------------------------
    // Debug: cycle-by-cycle monitoring (synthesis ignores)
    //-------------------------------------------------------------------------
    // synthesis translate_off
    always @(posedge clk) begin
        if (enable && !clear_acc)
            $display("[MAC] w=0x%04X (%6d)  a=0x%04X (%6d)  "
                     "p=0x%08X  acc=0x%08X  out=0x%04X (%6d)",
                     weight, weight, activation, activation,
                     product, accumulator, result, result);
    end
    // synthesis translate_on

endmodule
