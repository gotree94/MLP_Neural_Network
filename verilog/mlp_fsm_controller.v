//=============================================================================
// mlp_fsm_controller.v — FSM-based MLP Inference Controller
//=============================================================================
//
// This FSM controls the entire MLP inference pipeline:
//   1. Load input image (784 pixels)
//   2. Compute FC1 (784→64): 50,176 MACs
//   3. Apply bias + ReLU for FC1
//   4. Compute FC2 (64→32): 2,048 MACs
//   5. Apply bias + ReLU for FC2
//   6. Compute FC3 (32→10): 320 MACs
//   7. Apply bias for FC3 output
//   8. Argmax (find highest output class)
//   9. Assert done
//
// Architecture: 3-block FSM (State Register + Next State + Output)
// This is the recommended Xilinx coding style for FSM inference.
//
//=============================================================================

module mlp_fsm_controller #(
    parameter FC1_MAC_COUNT = 784 * 64,    // = 50176
    parameter FC2_MAC_COUNT = 64 * 32,     // = 2048
    parameter FC3_MAC_COUNT = 32 * 10,     // = 320
    parameter COUNTER_WIDTH = 20           // enough for 50176
) (
    input  wire         clk,
    input  wire         rst_n,
    // Control
    input  wire         start,              // pulse to start inference
    output reg          done,               // asserted when inference complete
    // FSM-driven control signals
    output reg          load_input,          // load input image from interface
    output reg          mac_enable,          // enable MAC operation
    output reg          mac_clear,           // clear MAC accumulator
    output reg          bias_enable,         // enable bias addition
    output reg          relu_enable,         // enable ReLU
    output reg          store_result,        // store layer result to buffer
    output reg          argmax_enable,       // enable argmax
    output reg  [1:0]   layer_select,        // 0:FC1, 1:FC2, 2:FC3, 3:OUT
    output wire [COUNTER_WIDTH-1:0] mac_addr, // address for weight BRAM
    output reg  [3:0]   state_debug          // for ILA debug
);

    //=========================================================================
    // State Encoding
    //=========================================================================
    // Using explicit binary encoding for clarity.
    // Vivado can re-encode to one-hot if it improves timing.
    localparam [3:0]
        S_IDLE      = 4'd0,
        S_LOAD_IN   = 4'd1,
        S_MAC_FC1   = 4'd2,
        S_BIAS_REL1 = 4'd3,
        S_MAC_FC2   = 4'd4,
        S_BIAS_REL2 = 4'd5,
        S_MAC_FC3   = 4'd6,
        S_BIAS_OUT  = 4'd7,
        S_ARGMAX    = 4'd8,
        S_DONE      = 4'd9;

    //=========================================================================
    // State Register (sequential)
    //=========================================================================
    reg [3:0] state, next_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    //=========================================================================
    // Next State Logic (combinational)
    //=========================================================================
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:      if (start)         next_state = S_LOAD_IN;
            S_LOAD_IN:                      next_state = S_MAC_FC1;
            S_MAC_FC1:   if (mac_done)      next_state = S_BIAS_REL1;
            S_BIAS_REL1:                    next_state = S_MAC_FC2;
            S_MAC_FC2:   if (mac_done)      next_state = S_BIAS_REL2;
            S_BIAS_REL2:                    next_state = S_MAC_FC3;
            S_MAC_FC3:   if (mac_done)      next_state = S_BIAS_OUT;
            S_BIAS_OUT:                     next_state = S_ARGMAX;
            S_ARGMAX:                       next_state = S_DONE;
            S_DONE:      if (!start)        next_state = S_IDLE;
            default:                        next_state = S_IDLE;
        endcase
    end

    //=========================================================================
    // MAC Address Counter
    //=========================================================================
    // Generates sequential addresses for weight BRAM.
    // Resets at the start of each layer, increments per MAC.

    reg [COUNTER_WIDTH-1:0] mac_counter;
    wire mac_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_counter <= 0;
        end else begin
            case (state)
                S_LOAD_IN:   mac_counter <= 0;
                S_MAC_FC1:   mac_counter <= (mac_counter < FC1_MAC_COUNT - 1)
                                            ? mac_counter + 1 : mac_counter;
                S_MAC_FC2:   mac_counter <= (mac_counter < FC2_MAC_COUNT - 1)
                                            ? mac_counter + 1 : mac_counter;
                S_MAC_FC3:   mac_counter <= (mac_counter < FC3_MAC_COUNT - 1)
                                            ? mac_counter + 1 : mac_counter;
                default:     mac_counter <= 0;
            endcase
        end
    end

    // MAC completion per layer
    assign mac_done =
        (state == S_MAC_FC1 && mac_counter >= FC1_MAC_COUNT - 1) ||
        (state == S_MAC_FC2 && mac_counter >= FC2_MAC_COUNT - 1) ||
        (state == S_MAC_FC3 && mac_counter >= FC3_MAC_COUNT - 1);

    assign mac_addr = mac_counter;

    //=========================================================================
    // Output Logic (combinational or registered)
    //=========================================================================

    // done signal (registered)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done <= 1'b0;
        else
            done <= (state == S_DONE);
    end

    // Control signals (combinational — decoded from current state)
    always @(*) begin
        // Defaults
        load_input    = 1'b0;
        mac_enable    = 1'b0;
        mac_clear     = 1'b0;
        bias_enable   = 1'b0;
        relu_enable   = 1'b0;
        store_result  = 1'b0;
        argmax_enable = 1'b0;
        layer_select  = 2'b00;

        case (state)
            S_LOAD_IN: begin
                load_input = 1'b1;
                mac_clear  = 1'b1;  // reset accumulator before layer
            end

            S_MAC_FC1: begin
                mac_enable   = 1'b1;
                layer_select = 2'b00;
            end

            S_BIAS_REL1: begin
                bias_enable  = 1'b1;
                relu_enable  = 1'b1;
                store_result = 1'b1;
                layer_select = 2'b00;
                mac_clear    = 1'b1;  // reset for next layer
            end

            S_MAC_FC2: begin
                mac_enable   = 1'b1;
                layer_select = 2'b01;
            end

            S_BIAS_REL2: begin
                bias_enable  = 1'b1;
                relu_enable  = 1'b1;
                store_result = 1'b1;
                layer_select = 2'b01;
                mac_clear    = 1'b1;
            end

            S_MAC_FC3: begin
                mac_enable   = 1'b1;
                layer_select = 2'b10;
            end

            S_BIAS_OUT: begin
                bias_enable  = 1'b1;
                store_result = 1'b1;
                layer_select = 2'b10;
                mac_clear    = 1'b1;
            end

            S_ARGMAX: begin
                argmax_enable = 1'b1;
                layer_select  = 2'b11;
            end

            default: ;
        endcase
    end

    // Debug output
    assign state_debug = state;

    //=========================================================================
    // Assertions (simulation only)
    //=========================================================================
    // synthesis translate_off
    always @(posedge clk) begin
        if (state == S_MAC_FC1 && mac_counter >= FC1_MAC_COUNT)
            $error("[FSM] FC1 MAC counter overflow! (%0d >= %0d)",
                   mac_counter, FC1_MAC_COUNT);
        if (state == S_MAC_FC2 && mac_counter >= FC2_MAC_COUNT)
            $error("[FSM] FC2 MAC counter overflow! (%0d >= %0d)",
                   mac_counter, FC2_MAC_COUNT);
        if (state == S_MAC_FC3 && mac_counter >= FC3_MAC_COUNT)
            $error("[FSM] FC3 MAC counter overflow! (%0d >= %0d)",
                   mac_counter, FC3_MAC_COUNT);
    end
    // synthesis translate_on

endmodule
