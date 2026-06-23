//=============================================================================
// relu.v — ReLU Activation Function
//=============================================================================
//
// ReLU(x) = max(0, x)
//
// In hardware this is trivial: simply check the MSB (sign bit).
// If MSB=1 (negative), output 0.
// If MSB=0 (positive or zero), output x unchanged.
//
// No DSP or BRAM needed — pure LUT logic.
//
//=============================================================================

module relu #(
    parameter DATA_WIDTH = 16
) (
    input  wire signed [DATA_WIDTH-1:0] data_in,   // Q8.8 signed
    output reg  signed [DATA_WIDTH-1:0] data_out   // Q8.8 signed
);

    always @(*) begin
        if (data_in[DATA_WIDTH-1])  // MSB = 1 → negative number
            data_out <= {DATA_WIDTH{1'b0}};  // output 0
        else
            data_out <= data_in;             // passthrough
    end

endmodule
