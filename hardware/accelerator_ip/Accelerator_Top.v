//============================================================================
// Accelerator_Top.v  -  SoC integration wrapper for the systolic-array accelerator
//----------------------------------------------------------------------------
// Port list matches the VHDL component `Accelerator_Top` in DVCon_SoC_SRC/TOP/Top.vhd
// EXACTLY, so Top.vhd needs no modification. This wrapper adapts the SoC's
// 64-bit full-AXI4 interfaces to our accelerator (top_axi_int):
//
//   SoC s_axi (AXI4, 64b data, 64b addr, 12b id)  --[axi_axil_adapter]-->  AXI-Lite (32b)  --> top_axi_int.s_axil
//   top_axi_int.m_axi (AXI4, 64b data, 64b addr, 12b id)  ------ direct ------>  SoC m_axi
//
// top_axi_int is reparameterised (config.svh: AXI_WIDTH=64, ADDR_WIDTH=64;
// AXI_ID_WIDTH=12; AXIL_BASE_ADDR=0x2004_0000) so its master matches the SoC
// memory bus natively. DDR lives at 0x8000_0000-0xBFFF_FFFF (fits the bus).
//============================================================================
`timescale 1ns/1ps

module Accelerator_Top (
    input  wire         s_axi_aclk,
    input  wire         s_axi_aresetn,

    // ---------------- AXI4 master (memory) : accelerator drives ----------------
    output wire         m_axi_awvalid,
    output wire [11:0]  m_axi_awid,
    output wire [7:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire [0:0]   m_axi_awlock,
    output wire [3:0]   m_axi_awcache,
    output wire [3:0]   m_axi_awqos,
    output wire [63:0]  m_axi_awaddr,
    output wire [2:0]   m_axi_awprot,
    input  wire         m_axi_awready,
    output wire         m_axi_wvalid,
    output wire         m_axi_wlast,
    output wire [63:0]  m_axi_wdata,
    output wire [7:0]   m_axi_wstrb,
    input  wire         m_axi_wready,
    output wire         m_axi_bready,
    input  wire         m_axi_bvalid,
    input  wire [11:0]  m_axi_bid,
    input  wire [1:0]   m_axi_bresp,
    output wire         m_axi_arvalid,
    output wire [11:0]  m_axi_arid,
    output wire [7:0]   m_axi_arlen,
    output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst,
    output wire [0:0]   m_axi_arlock,
    output wire [3:0]   m_axi_arcache,
    output wire [3:0]   m_axi_arqos,
    output wire [63:0]  m_axi_araddr,
    output wire [2:0]   m_axi_arprot,
    input  wire         m_axi_arready,
    output wire         m_axi_rready,
    input  wire         m_axi_rvalid,
    input  wire [11:0]  m_axi_rid,
    input  wire         m_axi_rlast,
    input  wire [1:0]   m_axi_rresp,
    input  wire [63:0]  m_axi_rdata,

    // ---------------- AXI4 slave (config) : SoC drives ----------------
    input  wire [11:0]  s_axi_awid,
    input  wire [63:0]  s_axi_awaddr,
    input  wire [7:0]   s_axi_awlen,
    input  wire [2:0]   s_axi_awsize,
    input  wire [1:0]   s_axi_awburst,
    input  wire         s_axi_awlock,
    input  wire [3:0]   s_axi_awcache,
    input  wire [2:0]   s_axi_awprot,
    input  wire [3:0]   s_axi_awqos,
    input  wire         s_axi_awvalid,
    output wire         s_axi_awready,
    input  wire [63:0]  s_axi_wdata,
    input  wire [7:0]   s_axi_wstrb,
    input  wire         s_axi_wlast,
    input  wire         s_axi_wvalid,
    output wire         s_axi_wready,
    input  wire         s_axi_bready,
    output wire [11:0]  s_axi_bid,
    output wire [1:0]   s_axi_bresp,
    output wire         s_axi_bvalid,
    input  wire [11:0]  s_axi_arid,
    input  wire [63:0]  s_axi_araddr,
    input  wire [7:0]   s_axi_arlen,
    input  wire [2:0]   s_axi_arsize,
    input  wire [1:0]   s_axi_arburst,
    input  wire         s_axi_arlock,
    input  wire [3:0]   s_axi_arcache,
    input  wire [2:0]   s_axi_arprot,
    input  wire [3:0]   s_axi_arqos,
    input  wire         s_axi_arvalid,
    output wire         s_axi_arready,
    input  wire         s_axi_rready,
    output wire [11:0]  s_axi_rid,
    output wire [63:0]  s_axi_rdata,
    output wire [1:0]   s_axi_rresp,
    output wire         s_axi_rlast,
    output wire         s_axi_rvalid
);

    // adapters use active-high reset; the accelerator uses active-low rstn
    wire rst_h = ~s_axi_aresetn;

    // AXI-Lite link between the config bridge and the accelerator (32b data, 64b addr)
    wire [63:0] axil_awaddr;  wire [2:0] axil_awprot;  wire axil_awvalid, axil_awready;
    wire [31:0] axil_wdata;   wire [3:0] axil_wstrb;   wire axil_wvalid,  axil_wready;
    wire [1:0]  axil_bresp;   wire axil_bvalid, axil_bready;
    wire [63:0] axil_araddr;  wire [2:0] axil_arprot;  wire axil_arvalid, axil_arready;
    wire [31:0] axil_rdata;   wire [1:0] axil_rresp;   wire axil_rvalid,  axil_rready;

    //----------------------------------------------------------------------
    // Config bridge: SoC full AXI4 (64b) -> AXI4-Lite (32b) for the registers
    //----------------------------------------------------------------------
    axi_axil_adapter #(
        .ADDR_WIDTH      (64),
        .AXI_DATA_WIDTH  (64),
        .AXI_ID_WIDTH    (12),
        .AXIL_DATA_WIDTH (32)
    ) u_cfg_bridge (
        .clk (s_axi_aclk),
        .rst (rst_h),
        // AXI4 slave (from SoC)
        .s_axi_awid(s_axi_awid),   .s_axi_awaddr(s_axi_awaddr), .s_axi_awlen(s_axi_awlen),
        .s_axi_awsize(s_axi_awsize),.s_axi_awburst(s_axi_awburst),.s_axi_awlock(s_axi_awlock),
        .s_axi_awcache(s_axi_awcache),.s_axi_awprot(s_axi_awprot),.s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wlast(s_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),.s_axi_wready(s_axi_wready),
        .s_axi_bid(s_axi_bid),     .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_arid(s_axi_arid),   .s_axi_araddr(s_axi_araddr), .s_axi_arlen(s_axi_arlen),
        .s_axi_arsize(s_axi_arsize),.s_axi_arburst(s_axi_arburst),.s_axi_arlock(s_axi_arlock),
        .s_axi_arcache(s_axi_arcache),.s_axi_arprot(s_axi_arprot),.s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rid(s_axi_rid),     .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rlast(s_axi_rlast), .s_axi_rvalid(s_axi_rvalid),.s_axi_rready(s_axi_rready),
        // AXI-Lite master (to accelerator)
        .m_axil_awaddr(axil_awaddr), .m_axil_awprot(axil_awprot), .m_axil_awvalid(axil_awvalid),
        .m_axil_awready(axil_awready),
        .m_axil_wdata(axil_wdata),   .m_axil_wstrb(axil_wstrb),   .m_axil_wvalid(axil_wvalid),
        .m_axil_wready(axil_wready),
        .m_axil_bresp(axil_bresp),   .m_axil_bvalid(axil_bvalid), .m_axil_bready(axil_bready),
        .m_axil_araddr(axil_araddr), .m_axil_arprot(axil_arprot), .m_axil_arvalid(axil_arvalid),
        .m_axil_arready(axil_arready),
        .m_axil_rdata(axil_rdata),   .m_axil_rresp(axil_rresp),   .m_axil_rvalid(axil_rvalid),
        .m_axil_rready(axil_rready)
    );

    //----------------------------------------------------------------------
    // The accelerator: AXI-Lite slave (from bridge) + 64-bit AXI4 master (to SoC)
    //----------------------------------------------------------------------
    top_axi_int u_accel (
        .clk  (s_axi_aclk),
        .rstn (s_axi_aresetn),
        // AXI-Lite slave (config)
        .s_axil_awaddr(axil_awaddr), .s_axil_awprot(axil_awprot), .s_axil_awvalid(axil_awvalid),
        .s_axil_awready(axil_awready),
        .s_axil_wdata(axil_wdata),   .s_axil_wstrb(axil_wstrb),   .s_axil_wvalid(axil_wvalid),
        .s_axil_wready(axil_wready),
        .s_axil_bresp(axil_bresp),   .s_axil_bvalid(axil_bvalid), .s_axil_bready(axil_bready),
        .s_axil_araddr(axil_araddr), .s_axil_arprot(axil_arprot), .s_axil_arvalid(axil_arvalid),
        .s_axil_arready(axil_arready),
        .s_axil_rdata(axil_rdata),   .s_axil_rresp(axil_rresp),   .s_axil_rvalid(axil_rvalid),
        .s_axil_rready(axil_rready),
        // AXI4 master (memory) - direct to SoC 64-bit bus
        .m_axi_awid(m_axi_awid),     .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),.m_axi_awlock(m_axi_awlock),
        .m_axi_awcache(m_axi_awcache),.m_axi_awprot(m_axi_awprot), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),   .m_axi_wstrb(m_axi_wstrb),   .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),       .m_axi_bresp(m_axi_bresp),   .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),     .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),.m_axi_arlock(m_axi_arlock),
        .m_axi_arcache(m_axi_arcache),.m_axi_arprot(m_axi_arprot), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),       .m_axi_rdata(m_axi_rdata),   .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),   .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );

    // The accelerator's master does not drive QoS; tie off (ignored by the memory slave)
    assign m_axi_awqos = 4'b0000;
    assign m_axi_arqos = 4'b0000;

endmodule
