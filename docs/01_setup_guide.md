# Module 0: Development Environment Setup

## 1. Install Xilinx Vivado ML Standard

1. Download Vivado ML Standard Edition from [AMD/Xilinx website](https://www.xilinx.com/support/download.html)
   - Select "Vivado ML Standard" (free, no license required for Zynq-7020)
   - Version: 2023.1 or later recommended

2. Run the installer:
   - Select "Vivado ML Standard" edition
   - Include: "Vivado Design Suite", "Vitis Unified IDE"
   - Devices: Select "7 Series" and "Zynq-7000"
   - Installation size: ~50 GB

3. Environment variables (Windows):
   - `C:\Xilinx\Vivado\2023.1\bin` should be in PATH
   - Verify: `vivado -version`

## 2. Install Zybo Z7-20 Board Files

1. Download from [Digilent GitHub](https://github.com/Digilent/vivado-board-files)
2. Extract to: `C:\Xilinx\Vivado\2023.1\data\boards\board_files`
3. Restart Vivado

## 3. First Project: LED Blink

```tcl
# Create a simple LED blink project to verify the toolchain
create_project -part xc7z020clg400-1 led_test ./led_test
set_property board_part digilentinc.com:zybo-z7-20:part0:1.0 [current_project]

# Create top-level Verilog
create_fileset -srcset sources_1
set fp [open ./led_test/src/led_blink.v w]
puts $fp "module led_blink(input clk, output reg led);"
puts $fp "  reg \[27:0\] counter;"
puts $fp "  always @(posedge clk) begin"
puts $fp "    counter <= counter + 1;"
puts $fp "    led <= counter\[27\];"
puts $fp "  end"
puts $fp "endmodule"
close $fp
add_files ./led_test/src/led_blink.v

# Create constraints
create_fileset -constrset constrs_1
set cfp [open ./led_test/src/zybo_pins.xdc w]
puts $cfp "set_property PACKAGE_PIN L16 \[get_ports clk\]"
puts $cfp "set_property IOSTANDARD LVCMOS33 \[get_ports clk\]"
puts $cfp "set_property PACKAGE_PIN M14 \[get_ports led\]"
puts $cfp "set_property IOSTANDARD LVCMOS33 \[get_ports led\]"
close $cfp
add_files ./led_test/src/zybo_pins.xdc

# Synthesize, implement, generate bitstream
launch_runs synth_1 -jobs 4; wait_on_run synth_1
launch_runs impl_1 -jobs 4; wait_on_run impl_1
open_run impl_1
write_bitstream -force ./led_test/led_blink.bit
```

## 4. Connect to Zybo Board

1. Connect USB-JTAG (micro-USB) to your PC
2. Connect USB power (or external 5V)
3. Open Vivado Hardware Manager → Auto Connect
4. Program the bitstream → LED should blink

## 5. UART Serial Console

- Baud rate: 115200 (default Zynq PS UART)
- Port: Check Device Manager → Ports (COM & LPT)
- Use PuTTY, TeraTerm, or `screen`

## Checklist

- [ ] Vivado installed and licensed
- [ ] Board files installed
- [ ] LED blink bitstream generated and programmed
- [ ] USB-JTAG communication working
- [ ] UART serial console working
