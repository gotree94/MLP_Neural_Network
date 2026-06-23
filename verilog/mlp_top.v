//=============================================================================
// mlp_top.v — MLP Inference Accelerator Top Module
//=============================================================================
//
// Complete 3-layer MLP (784→64→32→10) inference accelerator.
//
// Architecture:
//   - Single serial MAC (DSP48E1-based)
//   - FSM controller for layer sequencing
//   - BRAM for weight storage (one per layer)
//   - ReLU activation for hidden layers
//   - Argmax for output classification
//
// Interface:
//   - AXI4-Lite compatible control (start/done/predicted)
//   - Parallel pixel input (16-bit Q8.8)
//
// Resource Usage (estimated, Zynq-7020):
//   - DSP: 1 (MAC)
//   - BRAM: 3 (weight storage for FC1, FC2, FC3)
//   - LUT/FF: ~800 / ~400 (FSM + control logic)
//
//=============================================================================

module mlp_top #(
    parameter IMG_SIZE     = 784,
    parameter HIDDEN1_SIZE = 64,    // FC1 output
    parameter HIDDEN2_SIZE = 32,    // FC2 output
    parameter NUM_CLASSES  = 10,    // FC3 output
    parameter DATA_WIDTH   = 16     // Q8.8 fixed-point
) (
    // Clock and reset
    input  wire                     clk,
    input  wire                     rst_n,

    // Control interface (AXI4-Lite like)
    input  wire                     start,           // pulse to start
    output wire                     done,            // high when complete
    output wire [3:0]               predicted,       // predicted class (0-9)

    // Pixel input interface
    input  wire [DATA_WIDTH-1:0]    pixel_in,        // Q8.8 pixel value
    input  wire                     pixel_valid,
    output wire                     pixel_ready
);

    //=========================================================================
    // Parameter declarations
    //=========================================================================
    localparam FC1_WEIGHTS = IMG_SIZE * HIDDEN1_SIZE;  // 50176
    localparam FC2_WEIGHTS = HIDDEN1_SIZE * HIDDEN2_SIZE; // 2048
    localparam FC3_WEIGHTS = HIDDEN2_SIZE * NUM_CLASSES;  // 320

    localparam W_ADDR_WIDTH = 17;  // 2^17 = 131072 > 50176

    //=========================================================================
    // Signal declarations
    //=========================================================================
    // FSM control signals
    wire        load_input;
    wire        mac_enable;
    wire        mac_clear;
    wire        bias_enable;
    wire        relu_enable;
    wire        store_result;
    wire        argmax_enable;
    wire [1:0]  layer_select;
    wire [19:0] mac_addr;
    wire [3:0]  state_debug;

    // Datapath signals
    wire signed [DATA_WIDTH-1:0] weight_out;
    wire signed [DATA_WIDTH-1:0] mac_result;
    wire        mac_overflow;

    wire signed [DATA_WIDTH-1:0] bias_value;
    wire signed [DATA_WIDTH-1:0] biased_result;
    wire signed [DATA_WIDTH-1:0] activated_result;

    wire signed [DATA_WIDTH-1:0] layer_out [0:2];  // output buffers
    wire signed [DATA_WIDTH-1:0] fc3_outputs [0:NUM_CLASSES-1];

    // Addresses for each layer's weight BRAM
    wire [W_ADDR_WIDTH-1:0] addr_fc1;
    wire [W_ADDR_WIDTH-1:0] addr_fc2;
    wire [W_ADDR_WIDTH-1:0] addr_fc3;

    // Pixel input counter
    reg [9:0] pixel_count;  // 0..783

    // Internal result registers
    reg [3:0] predicted_reg;
    wire [3:0] argmax_class;
    wire argmax_valid;

    //=========================================================================
    // FSM Controller
    //=========================================================================
    mlp_fsm_controller #(
        .FC1_MAC_COUNT(FC1_WEIGHTS),
        .FC2_MAC_COUNT(FC2_WEIGHTS),
        .FC3_MAC_COUNT(FC3_WEIGHTS),
        .COUNTER_WIDTH(20)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done),
        .load_input(load_input),
        .mac_enable(mac_enable),
        .mac_clear(mac_clear),
        .bias_enable(bias_enable),
        .relu_enable(relu_enable),
        .store_result(store_result),
        .argmax_enable(argmax_enable),
        .layer_select(layer_select),
        .mac_addr(mac_addr),
        .state_debug(state_debug)
    );

    //=========================================================================
    // Weight BRAMs (one per layer)
    //=========================================================================
    // Address mapping: each layer's weight matrix is flattened row-major.
    // FC1: weight[hidden_idx][input_idx] → addr = hidden_idx*784 + input_idx
    // FC2: weight[hidden_idx][input_idx] → addr = hidden_idx*64 + input_idx
    // FC3: weight[class_idx][hidden_idx] → addr = class_idx*32 + hidden_idx

    assign addr_fc1 = mac_addr;
    assign addr_fc2 = mac_addr;
    assign addr_fc3 = mac_addr;

    bram_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(W_ADDR_WIDTH),
        .INIT_PARAM("FC1_WEIGHT_INIT")
    ) u_bram_fc1 (
        .clk(clk),
        .en(mac_enable && (layer_select == 2'b00)),
        .addr(addr_fc1),
        .dout()
    );

    bram_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(W_ADDR_WIDTH),
        .INIT_PARAM("FC2_WEIGHT_INIT")
    ) u_bram_fc2 (
        .clk(clk),
        .en(mac_enable && (layer_select == 2'b01)),
        .addr(addr_fc2),
        .dout()
    );

    bram_wrapper #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(W_ADDR_WIDTH),
        .INIT_PARAM("FC3_WEIGHT_INIT")
    ) u_bram_fc3 (
        .clk(clk),
        .en(mac_enable && (layer_select == 2'b10)),
        .addr(addr_fc3),
        .dout()
    );

    // Weight mux: select weight from the active layer's BRAM
    reg signed [DATA_WIDTH-1:0] current_weight;
    always @(*) begin
        case (layer_select)
            2'b00:    current_weight = ;      // FC1 weight
            2'b01:    current_weight = ;      // FC2 weight
            2'b10:    current_weight = ;      // FC3 weight
            default:  current_weight = 0;
        endcase
    end

    //=========================================================================
    // Activation mux: select input for each layer
    //=========================================================================
    // FC1: pixels from input buffer
    // FC2: output of FC1 (after ReLU)
    // FC3: output of FC2 (after ReLU)

    reg signed [DATA_WIDTH-1:0] current_activation;
    reg signed [DATA_WIDTH-1:0] pixel_buffer [0:IMG_SIZE-1];
    reg signed [DATA_WIDTH-1:0] layer1_buffer [0:HIDDEN1_SIZE-1];
    reg signed [DATA_WIDTH-1:0] layer2_buffer [0:HIDDEN2_SIZE-1];

    // Address counters per layer (which neuron we're computing)
    wire [9:0] pixel_idx;   // 0..783
    wire [6:0] hidden1_idx; // 0..63
    wire [5:0] hidden2_idx; // 0..31
    wire [3:0] class_idx;   // 0..9

    assign pixel_idx   = mac_addr % IMG_SIZE;
    assign hidden1_idx = mac_addr / IMG_SIZE;
    assign hidden2_idx = mac_addr / HIDDEN1_SIZE;
    assign class_idx   = mac_addr / HIDDEN2_SIZE;

    always @(*) begin
        case (layer_select)
            2'b00: current_activation = pixel_buffer[pixel_idx];
            2'b01: current_activation = layer1_buffer[mac_addr % HIDDEN1_SIZE];
            2'b10: current_activation = layer2_buffer[mac_addr % HIDDEN2_SIZE];
            default: current_activation = 0;
        endcase
    end

    //=========================================================================
    // MAC Unit
    //=========================================================================
    mac_unit #(.DATA_WIDTH(DATA_WIDTH)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .enable(mac_enable),
        .clear_acc(mac_clear),
        .weight(current_weight),
        .activation(current_activation),
        .result(mac_result),
        .overflow(mac_overflow)
    );

    //=========================================================================
    // Bias Addition
    //=========================================================================
    // Biases are stored as localparam arrays (included via `include)
    // Bias address = current output neuron index within the layer

    `include "../coe/fc1_bias.vh"
    `include "../coe/fc2_bias.vh"
    `include "../coe/fc3_bias.vh"

    reg [6:0] neuron_idx;  // which neuron we're computing

    always @(*) begin
        case (layer_select)
            2'b00: neuron_idx = mac_addr / IMG_SIZE;         // FC1: 0..63
            2'b01: neuron_idx = mac_addr / HIDDEN1_SIZE;     // FC2: 0..31
            2'b10: neuron_idx = mac_addr / HIDDEN2_SIZE;     // FC3: 0..9
            default: neuron_idx = 0;
        endcase
    end

    always @(*) begin
        case (layer_select)
            2'b00: bias_value = FC1_BIAS[neuron_idx];
            2'b01: bias_value = FC2_BIAS[neuron_idx];
            2'b10: bias_value = FC3_BIAS[neuron_idx];
            default: bias_value = 0;
        endcase
    end

    // Saturated bias addition
    wire signed [DATA_WIDTH:0] bias_sum;
    assign bias_sum = {mac_result[DATA_WIDTH-1], mac_result} +
                      {bias_value[DATA_WIDTH-1], bias_value};

    always @(*) begin
        if (bias_sum > 16'h7FFF)
            biased_result <= 16'h7FFF;
        else if (bias_sum < 16'h8000)
            biased_result <= 16'h8000;
        else
            biased_result <= bias_sum[DATA_WIDTH-1:0];
    end

    //=========================================================================
    // ReLU Activation
    //=========================================================================
    relu #(.DATA_WIDTH(DATA_WIDTH)) u_relu (
        .data_in(biased_result),
        .data_out(activated_result)
    );

    wire signed [DATA_WIDTH-1:0] final_result;
    assign final_result = relu_enable ? activated_result : biased_result;

    //=========================================================================
    // Output Buffer Storage
    //=========================================================================
    // Store MAC results to the appropriate layer buffer when store_result=1

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < IMG_SIZE; i = i + 1)
                pixel_buffer[i] <= 0;
            for (int i = 0; i < HIDDEN1_SIZE; i = i + 1)
                layer1_buffer[i] <= 0;
            for (int i = 0; i < HIDDEN2_SIZE; i = i + 1)
                layer2_buffer[i] <= 0;
            for (int i = 0; i < NUM_CLASSES; i = i + 1)
                fc3_outputs[i] <= 0;
        end else begin
            // Load input pixels
            if (load_input && pixel_valid) begin
                if (pixel_count < IMG_SIZE) begin
                    pixel_buffer[pixel_count] <= pixel_in;
                    pixel_count <= pixel_count + 1;
                end
            end

            // Store layer results after bias+ReLU
            if (store_result) begin
                case (layer_select)
                    2'b00: layer1_buffer[neuron_idx] <= final_result;
                    2'b01: layer2_buffer[neuron_idx] <= final_result;
                    2'b10: fc3_outputs[neuron_idx] <= final_result;
                endcase
            end

            // Reset pixel counter when done loading
            if (load_input && pixel_count >= IMG_SIZE)
                pixel_count <= 0;
        end
    end

    assign pixel_ready = load_input && (pixel_count < IMG_SIZE);

    //=========================================================================
    // Argmax (Output Classification)
    //=========================================================================
    argmax #(
        .NUM_CLASSES(NUM_CLASSES),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_argmax (
        .clk(clk), .rst_n(rst_n),
        .enable(argmax_enable),
        .class_scores(fc3_outputs),
        .predicted_class(argmax_class),
        .valid(argmax_valid)
    );

    //=========================================================================
    // Output Register
    //=========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            predicted_reg <= 0;
        else if (argmax_valid)
            predicted_reg <= argmax_class;
    end

    assign predicted = predicted_reg;

endmodule
