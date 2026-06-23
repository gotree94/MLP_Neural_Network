#=============================================================================
# create_project.tcl — Vivado Project Creation Script
#=============================================================================
#
# This Tcl script automates the creation of a Vivado project for the
# MLP Neural Network Accelerator targeting the Zybo Z7-20 board.
#
# Usage:
#   vivado -mode tcl -source create_project.tcl
#
# Or from within Vivado Tcl Console:
#   source C:/path/to/create_project.tcl
#
#=============================================================================

#-------------------------------------------------------------------------
# Step 1: Project Settings
#-------------------------------------------------------------------------
set project_name "mlp_zybo_accelerator"
set project_dir  "./${project_name}"
set part_number  "xc7z020clg400-1"
set board_part   "digilentinc.com:zybo-z7-20:part0:1.0"

puts "============================================"
puts "MLP Accelerator — Vivado Project Creation"
puts "============================================"
puts "Project: ${project_name}"
puts "Part:    ${part_number}"
puts "Board:   ${board_part}"

#-------------------------------------------------------------------------
# Step 2: Create Project
#-------------------------------------------------------------------------
create_project -part ${part_number} -force ${project_name} ${project_dir}
set_property board_part ${board_part} [current_project]
set_property target_language Verilog [current_project]
set_property simulator_language Verilog [current_project]

puts "  [OK] Project created."

#-------------------------------------------------------------------------
# Step 3: Create Filesets
#-------------------------------------------------------------------------
# Design sources
if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}

# Simulation sources
if {[string equal [get_filesets -quiet sim_1] ""]} {
    create_fileset -simset sim_1
}

# Constraints
if {[string equal [get_filesets -quiet constrs_1] ""]} {
    create_fileset -constrset constrs_1
}

puts "  [OK] Filesets created."

#-------------------------------------------------------------------------
# Step 4: Add RTL Source Files
#-------------------------------------------------------------------------
# Adjust these paths to match your actual file locations
set rtl_sources [list \
    "../verilog/mac_unit.v" \
    "../verilog/mlp_fsm_controller.v" \
    "../verilog/relu.v" \
    "../verilog/argmax.v" \
    "../verilog/bram_wrapper.v" \
    "../verilog/mlp_top.v" \
]

puts "  Adding RTL sources..."
foreach src $rtl_sources {
    if {[file exists $src]} {
        add_files -norecurse -fileset [get_filesets sources_1] $src
        puts "    + $src"
    } else {
        puts "    ! WARNING: $src not found (skipping)"
    }
}

#-------------------------------------------------------------------------
# Step 5: Add Testbench Files
#-------------------------------------------------------------------------
set tb_sources [list \
    "../verilog/tb_mlp_top.v" \
]

puts "  Adding testbench sources..."
foreach src $tb_sources {
    if {[file exists $src]} {
        add_files -norecurse -fileset [get_filesets sim_1] $src
        puts "    + $src"
    }
}

# Set tb_mlp_top as the top-level simulation module
set_property top tb_mlp_top [get_filesets sim_1]

#-------------------------------------------------------------------------
# Step 6: Add Constraints
#-------------------------------------------------------------------------
# Create a basic Zybo Z7-20 constraint file
set constr_file [file join $project_dir "src" "constrs" "zybo_z7_20.xdc"]

# Create the constraints directory
file mkdir [file dirname $constr_file]

# Write a minimal constraint file
set constr_fp [open $constr_file "w"]
puts $constr_fp "# Zybo Z7-20 Master Pin Constraints"
puts $constr_fp "# Generated for MLP Accelerator Project"
puts $constr_fp ""
puts $constr_fp "# 100 MHz System Clock"
puts $constr_fp "create_clock -period 10.000 -name sys_clk -waveform {0 5} \[get_ports {clk}\]"
puts $constr_fp "set_property PACKAGE_PIN L16 \[get_ports {clk}\]"
puts $constr_fp "set_property IOSTANDARD LVCMOS33 \[get_ports {clk}\]"
puts $constr_fp ""
puts $constr_fp "# Buttons (active low)"
puts $constr_fp "set_property PACKAGE_PIN N15 \[get_ports {rst_n}\]"
puts $constr_fp "set_property IOSTANDARD LVCMOS33 \[get_ports {rst_n}\]"
puts $constr_fp ""
puts $constr_fp "# LEDs for result display"
puts $constr_fp "set_property PACKAGE_PIN M14 \[get_ports {predicted\[0\]}\]"
puts $constr_fp "set_property PACKAGE_PIN M15 \[get_ports {predicted\[1\]}\]"
puts $constr_fp "set_property PACKAGE_PIN G14 \[get_ports {predicted\[2\]}\]"
puts $constr_fp "set_property PACKAGE_PIN D18 \[get_ports {predicted\[3\]}\]"
puts $constr_fp "set_property IOSTANDARD LVCMOS33 \[get_ports {predicted\[*\]}\]"
puts $constr_fp ""
puts $constr_fp "# Done LED"
puts $constr_fp "set_property PACKAGE_PIN J15 \[get_ports {done}\]"
puts $constr_fp "set_property IOSTANDARD LVCMOS33 \[get_ports {done}\]"
close $constr_fp

add_files -norecurse -fileset [get_filesets constrs_1] $constr_file
puts "  [OK] Constraints added."

#-------------------------------------------------------------------------
# Step 7: Set Top Module
#-------------------------------------------------------------------------
set_property top mlp_top [current_fileset]
puts "  [OK] Top module set to mlp_top."

#-------------------------------------------------------------------------
# Step 8: Project Properties
#-------------------------------------------------------------------------
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property strategy Performance_Explore [get_runs impl_1]

puts ""
puts "============================================"
puts "Project creation complete!"
puts "============================================"
puts ""
puts "Next steps:"
puts "  1. In Vivado: Tools → Run Synthesis"
puts "  2. Tools → Run Implementation"
puts "  3. Tools → Generate Bitstream"
puts "  4. File → Export → Export Hardware (include bitstream)"
puts "  5. Launch Vitis to build ARM firmware"
puts ""
puts "Or run all at once:"
puts "  launch_runs synth_1 -jobs 4"
puts "  wait_on_run synth_1"
puts "  launch_runs impl_1 -jobs 4"
puts "  wait_on_run impl_1"
puts "  open_run impl_1"
puts "  write_bitstream -force ${project_dir}/${project_name}.bit"
puts "============================================"

#-------------------------------------------------------------------------
# Optional: Run synthesis immediately (uncomment to enable)
#-------------------------------------------------------------------------
# puts "Running synthesis..."
# launch_runs synth_1 -jobs 4
# wait_on_run synth_1

# puts "Running implementation..."
# launch_runs impl_1 -jobs 4
# wait_on_run impl_1

# puts "Generating bitstream..."
# open_run impl_1
# write_bitstream -force "${project_dir}/${project_name}.bit"
# puts "Bitstream: ${project_dir}/${project_name}.bit"
