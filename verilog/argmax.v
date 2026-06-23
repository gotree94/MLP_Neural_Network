//=============================================================================
// argmax.v — Argmax: Find Index of Maximum Value
//=============================================================================
//
// Takes an array of NUM_CLASSES scores and returns the index of the
// highest value. This converts the 10-element output vector into
// a single digit prediction (0-9).
//
// Architecture: sequential comparison
//   - Iterate through all 10 scores
//   - Track the maximum value and its index
//   - Output the index when done
//
// Timing: 10 clock cycles per inference (one per class)
//
//=============================================================================

module argmax #(
    parameter NUM_CLASSES = 10,
    parameter DATA_WIDTH  = 16
) (
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             enable,     // start argmax
    input  wire signed [DATA_WIDTH-1:0]     class_scores [0:NUM_CLASSES-1],
    output reg  [3:0]                       predicted_class,
    output reg                              valid        // result ready
);

    // State
    reg [3:0] idx_counter;
    reg signed [DATA_WIDTH-1:0] max_value;
    reg running;

    // Argmax FSM
    localparam IDLE = 1'b0;
    localparam RUN  = 1'b1;

    reg state, next_state;

    // State register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: if (enable)  next_state = RUN;
            RUN:  if (idx_counter >= NUM_CLASSES - 1) next_state = IDLE;
        endcase
    end

    // Counter and comparison
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idx_counter     <= 0;
            max_value       <= 0;
            predicted_class <= 0;
            valid           <= 0;
            running         <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable) begin
                        // Initialize with first class
                        idx_counter     <= 0;
                        max_value       <= class_scores[0];
                        predicted_class <= 0;
                        valid           <= 0;
                        running         <= 1;
                    end else begin
                        valid   <= 0;
                        running <= 0;
                    end
                end

                RUN: begin
                    idx_counter <= idx_counter + 1;

                    // Compare current score with max
                    if (class_scores[idx_counter + 1] > max_value) begin
                        max_value       <= class_scores[idx_counter + 1];
                        predicted_class <= idx_counter + 1;
                    end

                    // Last iteration?
                    if (idx_counter >= NUM_CLASSES - 2) begin
                        valid   <= 1;
                        running <= 0;
                    end
                end
            endcase
        end
    end

    // synthesis translate_off
    // Debug display
    always @(posedge clk) begin
        if (valid)
            $display("[ARGMAX] Predicted class: %0d (score=%6d)",
                     predicted_class, max_value);
    end
    // synthesis translate_on

endmodule
