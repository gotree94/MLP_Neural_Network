//=============================================================================
// tb_mlp_top.v — Testbench for MLP Top Module
//=============================================================================
//
// This testbench verifies the MLP accelerator against the Python-generated
// "Golden Reference" results.
//
// Test flow:
//   1. Load test image from hex file
//   2. Send pixels to the DUT
//   3. Assert start and wait for done
//   4. Compare prediction with expected label
//   5. Repeat for multiple test images
//   6. Print pass/fail summary
//
//=============================================================================

`timescale 1ns / 1ps

module tb_mlp_top();

    //=========================================================================
    // Parameters
    //=========================================================================
    localparam CLK_PERIOD   = 10;    // 100 MHz
    localparam IMG_SIZE     = 784;
    localparam NUM_IMAGES   = 5;     // number of test images
    localparam DATA_WIDTH   = 16;

    //=========================================================================
    // DUT Signals
    //=========================================================================
    reg                     clk;
    reg                     rst_n;
    reg                     start;
    reg  [DATA_WIDTH-1:0]   pixel_in;
    reg                     pixel_valid;
    wire                    pixel_ready;
    wire                    done;
    wire [3:0]              predicted;

    //=========================================================================
    // Test Data
    //=========================================================================
    reg [DATA_WIDTH-1:0] test_images [0:NUM_IMAGES-1][0:IMG_SIZE-1];
    reg [3:0]            test_labels [0:NUM_IMAGES-1];
    integer              pass_count, fail_count;

    //=========================================================================
    // DUT Instantiation
    //=========================================================================
    mlp_top #(
        .IMG_SIZE(784),
        .HIDDEN1_SIZE(64),
        .HIDDEN2_SIZE(32),
        .NUM_CLASSES(10),
        .DATA_WIDTH(16)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .predicted(predicted),
        .pixel_in(pixel_in),
        .pixel_valid(pixel_valid),
        .pixel_ready(pixel_ready)
    );

    //=========================================================================
    // Clock Generation
    //=========================================================================
    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    //=========================================================================
    // Test Procedure
    //=========================================================================
    initial begin
        integer img, px;

        $display("==========================================================");
        $display("MLP Inference Accelerator — Testbench");
        $display("Architecture: 784 → 64 → 32 → 10 (Q8.8 fixed-point)");
        $display("==========================================================\n");

        //-------------------------------------------------------------------------
        // Step 1: Load test data
        //-------------------------------------------------------------------------
        $display("[TB] Loading test data...");
        for (img = 0; img < NUM_IMAGES; img = img + 1) begin
            $readmemh($sformatf("../../coe/test_image_%0d.hex", img),
                      test_images[img]);

            // Read label from file
            // In practice, labels are embedded in golden_results.txt
            test_labels[img] = 0;  // placeholder — updated from golden file
        end

        // For this test, use hardcoded labels (from golden_results.txt)
        // These MUST match the Python-generated labels.
        // (In a real test, parse golden_results.txt instead.)
        test_labels[0] = 7;
        test_labels[1] = 2;
        test_labels[2] = 1;
        test_labels[3] = 0;
        test_labels[4] = 4;

        //-------------------------------------------------------------------------
        // Step 2: Reset
        //-------------------------------------------------------------------------
        $display("[TB] Resetting DUT...");
        rst_n       = 0;
        start       = 0;
        pixel_in    = 0;
        pixel_valid = 0;
        pass_count  = 0;
        fail_count  = 0;

        #100;
        rst_n = 1;
        #20;

        //-------------------------------------------------------------------------
        // Step 3: Run inference for each test image
        //-------------------------------------------------------------------------
        for (img = 0; img < NUM_IMAGES; img = img + 1) begin
            $display("\n[TB] Image %0d of %0d (expected label: %0d)",
                     img + 1, NUM_IMAGES, test_labels[img]);

            // Send pixels to DUT
            $display("[TB] Loading %0d pixels...", IMG_SIZE);
            for (px = 0; px < IMG_SIZE; px = px + 1) begin
                @(posedge clk);
                pixel_in    <= test_images[img][px];
                pixel_valid <= 1;
            end
            @(posedge clk);
            pixel_valid <= 0;

            // Assert start
            @(posedge clk);
            start <= 1;
            @(posedge clk);
            start <= 0;

            $display("[TB] Inference started...");

            // Wait for completion
            wait(done);
            #20;  // wait for output to settle

            // Check result
            if (predicted == test_labels[img]) begin
                $display("[TB] ✓ PASS: predicted=%0d, expected=%0d",
                         predicted, test_labels[img]);
                pass_count = pass_count + 1;
            end else begin
                $display("[TB] ✗ FAIL: predicted=%0d, expected=%0d",
                         predicted, test_labels[img]);
                fail_count = fail_count + 1;
            end

            // Wait a bit before next image
            #100;
        end

        //-------------------------------------------------------------------------
        // Step 4: Summary
        //-------------------------------------------------------------------------
        $display("\n==========================================================");
        $display("RESULTS: %0d PASS, %0d FAIL out of %0d tests",
                 pass_count, fail_count, NUM_IMAGES);

        if (fail_count == 0)
            $display("★★★ ALL TESTS PASSED ★★★");
        else
            $display("★★★ SOME TESTS FAILED ★★★");
        $display("==========================================================");

        //-------------------------------------------------------------------------
        // Step 5: Waveform dump for visualization
        //-------------------------------------------------------------------------
        $dumpfile("mlp_inference.vcd");
        $dumpvars(0, tb_mlp_top);

        #200;
        $finish;
    end

endmodule
