//=============================================================================
// bram_wrapper.v — BRAM Wrapper for Weight Storage
//=============================================================================
//
// Simple single-port BRAM wrapper that stores MLP weights.
// Initialized from a hex file via $readmemh for simulation.
// For synthesis, actual weights are embedded in the bitstream
// via Xilinx Block Memory Generator IP or INIT attribute.
//
// Interface:
//   - Synchronous read (registered output)
//   - Single cycle latency
//   - Combinational address decode
//
//=============================================================================

module bram_wrapper #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16,         // max address space
    parameter INIT_FILE  = "none",      // hex file for $readmemh
    parameter INIT_PARAM = "none"       // alternate init string
) (
    input  wire                     clk,
    input  wire                     en,          // read enable
    input  wire [ADDR_WIDTH-1:0]    addr,        // read address
    output reg  [DATA_WIDTH-1:0]    dout         // read data (registered)
);

    // BRAM storage
    (* ram_style = "block" *)  // force block RAM (not distributed)
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Initialize from hex file (simulation)
    initial begin
        if (INIT_FILE != "none") begin
            $readmemh(INIT_FILE, mem);
            $display("[BRAM] Initialized from %s", INIT_FILE);
        end else begin
            // Fill with zeros if no init file
            for (int i = 0; i < (1<<ADDR_WIDTH); i = i + 1)
                mem[i] = 0;
        end
    end

    // Synchronous read (registered output)
    always @(posedge clk) begin
        if (en)
            dout <= mem[addr];
    end

endmodule
