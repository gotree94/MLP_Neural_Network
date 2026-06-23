#=============================================================================
# block_design.tcl — Vivado Block Design for Zynq PS + MLP Accelerator
#=============================================================================
#
# Creates a Zynq Processing System + AXI GPIO interface to control
# the MLP accelerator from the ARM Cortex-A9 processor.
#
# Architecture:
#   ARM (PS) ──AXI─→ SmartConnect ──→ AXI GPIO (control)
#                                    ──→ AXI GPIO (status)
#                                    ──→ AXI GPIO (data out)
#
# Usage (after create_project.tcl):
#   source C:/path/to/block_design.tcl
#
#=============================================================================

#-------------------------------------------------------------------------
# Step 1: Create Block Design
#-------------------------------------------------------------------------
set design_name "mlp_system"
create_bd_design $design_name

puts "Creating block design: ${design_name}"

#-------------------------------------------------------------------------
# Step 2: Zynq Processing System
#-------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 \
    processing_system7_0

# Apply Zybo Z7-20 board preset
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" \
             apply_board_preset "1" \
             Master "Disable" \
             Slave "Disable" } \
    [get_bd_cells processing_system7_0]

puts "  [OK] Zynq PS configured with Zybo Z7-20 board preset."

#-------------------------------------------------------------------------
# Step 3: Enable AXI GPIO interrupts for done signal
#-------------------------------------------------------------------------
# Enable IRQ_F2P on the PS so the PL can signal completion
set_property -dict [list \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
] [get_bd_cells processing_system7_0]

#-------------------------------------------------------------------------
# Step 4: AXI GPIO — Control (PS → PL)
#-------------------------------------------------------------------------
# Channel 1: 2-bit output (start, rst)
# Channel 2: 16-bit output (pixel data)

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_ctrl
set_property -dict [list \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_GPIO_WIDTH {2} \
    CONFIG.C_ALL_OUTPUTS_1 {1} \
    CONFIG.C_GPIO_WIDTH_1 {16} \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_DOUT_DEFAULT_0 {0x00000000} \
] [get_bd_cells axi_gpio_ctrl]

puts "  [OK] AXI GPIO (control) created."

#-------------------------------------------------------------------------
# Step 5: AXI GPIO — Status (PL → PS)
#-------------------------------------------------------------------------
# Channel 1: 8-bit input (done, predicted_class[3:0], reserved)

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_status
set_property -dict [list \
    CONFIG.C_ALL_INPUTS {1} \
    CONFIG.C_GPIO_WIDTH {8} \
    CONFIG.C_INTERRUPT_PRESENT {1} \
] [get_bd_cells axi_gpio_status]

puts "  [OK] AXI GPIO (status) created."

#-------------------------------------------------------------------------
# Step 6: SmartConnect (AXI Interconnect)
#-------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] \
    [get_bd_cells smartconnect_0]

puts "  [OK] SmartConnect created."

#-------------------------------------------------------------------------
# Step 7: Processor System Reset
#-------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0

puts "  [OK] Processor System Reset created."

#-------------------------------------------------------------------------
# Step 8: MLP Accelerator (RTL Module)
#-------------------------------------------------------------------------
# This assumes mlp_top.v is already added to the project sources
create_bd_cell -type module -reference mlp_top mlp_top_0

puts "  [OK] MLP Accelerator (mlp_top) added to block design."

#-------------------------------------------------------------------------
# Step 9: AXI Connections
#-------------------------------------------------------------------------
# Connect PS M_AXI_GP0 → SmartConnect (master)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Clk_master {Auto} \
             Clk_slave {Auto} \
             Clk_xbar {Auto} \
             Master {/processing_system7_0/M_AXI_GP0} \
             Slave {/smartconnect_0/S00_AXI}} \
    [get_bd_intf_pins smartconnect_0/S00_AXI]

puts "  [OK] AXI bus connected."

# Connect SmartConnect → AXI GPIO slaves
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] \
    [get_bd_intf_pins axi_gpio_ctrl/S_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M01_AXI] \
    [get_bd_intf_pins axi_gpio_status/S_AXI]

#-------------------------------------------------------------------------
# Step 10: Clock and Reset Connections
#-------------------------------------------------------------------------
# PS clock → MLP accelerator
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins mlp_top_0/clk]
# PS clock → MLP accelerator
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins axi_gpio_ctrl/s_axi_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins axi_gpio_status/s_axi_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk]

# Reset
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins mlp_top_0/rst_n]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_gpio_ctrl/s_axi_aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_gpio_status/s_axi_aresetn]

# External reset
connect_bd_net [get_bd_pins proc_sys_reset_0/ext_reset_in] \
    [get_bd_pins processing_system7_0/FCLK_RESET0_N]

puts "  [OK] Clock and reset connected."

#-------------------------------------------------------------------------
# Step 11: Data Connections (GPIO → MLP Accelerator)
#-------------------------------------------------------------------------
# Control: GPIO channel 1 → MLP start/rst
connect_bd_net [get_bd_pins axi_gpio_ctrl/gpio_io_o[0]] \
    [get_bd_pins mlp_top_0/start]
# GPIO channel 2[15:0] → pixel_in
connect_bd_net [get_bd_pins axi_gpio_ctrl/gpio2_io_o] \
    [get_bd_pins mlp_top_0/pixel_in]

# Status: MLP done → GPIO bit 0
connect_bd_net [get_bd_pins mlp_top_0/done] \
    [get_bd_pins axi_gpio_status/gpio_io_i[0]]
# MLP predicted[3:0] → GPIO bits 4-7
connect_bd_net [get_bd_pins mlp_top_0/predicted[0]] \
    [get_bd_pins axi_gpio_status/gpio_io_i[4]]
connect_bd_net [get_bd_pins mlp_top_0/predicted[1]] \
    [get_bd_pins axi_gpio_status/gpio_io_i[5]]
connect_bd_net [get_bd_pins mlp_top_0/predicted[2]] \
    [get_bd_pins axi_gpio_status/gpio_io_i[6]]
connect_bd_net [get_bd_pins mlp_top_0/predicted[3]] \
    [get_bd_pins axi_gpio_status/gpio_io_i[7]]

puts "  [OK] Data connections established."

#-------------------------------------------------------------------------
# Step 12: External Ports
#-------------------------------------------------------------------------
make_bd_pins_external [get_bd_pins processing_system7_0/DDR]
make_bd_pins_external [get_bd_pins processing_system7_0/FIXED_IO]

puts "  [OK] External ports created."

#-------------------------------------------------------------------------
# Step 13: Validate and Generate
#-------------------------------------------------------------------------
validate_bd_design
puts "  [OK] Block design validated."

generate_target all [get_files */${design_name}.bd]
puts "  [OK] HDL wrapper and IP cores generated."

# Create HDL wrapper for the block design
make_wrapper -files [get_files */${design_name}.bd] -top
add_files -norecurse [glob "./*${design_name}_wrapper.v"]

puts ""
puts "============================================"
puts "Block design complete: ${design_name}"
puts "============================================"
puts ""
puts "Address map (AXI GPIO):"
puts "  axi_gpio_ctrl:  0x4120_0000 (channel1: start, channel2: pixel)"
puts "  axi_gpio_status: 0x4121_0000 (bit0: done, bits4-7: predicted)"
puts ""
puts "Review address assignments in:"
puts "  Address Editor tab → auto-assign addresses"
