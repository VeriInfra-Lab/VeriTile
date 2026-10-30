import bench.tritonbench_g.swiglu_backward.Sources.Mathematical_Kernels

/-! Complete ordinary FP32 source for the checked kernel and its retained
integer-address candidate. Both constexpr recomputation branches stay symbolic;
the generic canonical source remains available through `sourceKernel`. -/
namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SwigluBackward.typedKernel

/-- Original casts, sigmoid, multiplication order and three ordered masked stores. -/
def kernel (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy cols block : Nat) (recompute : Bool) : ComputeKernel :=
  triton {
    tl.ports inputs=[X,Y,DOUT], outputs=[DX,DY,OUT]
    tl.region X = tl.float32, Y = tl.float32, DOUT = tl.float32, DX = tl.float32, DY = tl.float32, OUT = tl.float32
    row = tl.program_id(0)
    start_col = tl.program_id(1) * $(block)
    X += row * $(sx)
    Y += row * $(sy)
    DOUT += row * $(sd)
    if recompute { OUT += row * $(so) }
    DX += row * $(sdx)
    DY += row * $(sdy)
    cols = start_col + tl.arange(0, $(block))
    x = tl.load(X + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < $(cols))
    tl.store(DY + cols, dy, mask=cols < $(cols))
    if recompute {
      out = x * x_sigmoid * y
      tl.store(OUT + cols, out, mask=cols < $(cols))
    }
  }

/-- Existing candidate: only the six natural row-stride products are commuted. -/
def commutedKernel (X Y DOUT OUT DX DY : RegionName)
    (sx sy sd so sdx sdy cols block : Nat) (recompute : Bool) : ComputeKernel :=
  triton {
    tl.ports inputs=[X,Y,DOUT], outputs=[DX,DY,OUT]
    tl.region X = tl.float32, Y = tl.float32, DOUT = tl.float32, DX = tl.float32, DY = tl.float32, OUT = tl.float32
    row = tl.program_id(0)
    start_col = tl.program_id(1) * $(block)
    X += $(sx) * row
    Y += $(sy) * row
    DOUT += $(sd) * row
    if recompute { OUT += $(so) * row }
    DX += $(sdx) * row
    DY += $(sdy) * row
    cols = start_col + tl.arange(0, $(block))
    x = tl.load(X + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < $(cols))
    tl.store(DY + cols, dy, mask=cols < $(cols))
    if recompute {
      out = x * x_sigmoid * y
      tl.store(OUT + cols, out, mask=cols < $(cols))
    }
  }

end VeriTile.Bench.Repairs.SwigluBackward
