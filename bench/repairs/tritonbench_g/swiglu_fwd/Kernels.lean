import bench.tritonbench_g.swiglu_fwd.Kernels

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton

/-- Historical independently typed storage family, retained for compatibility. -/
abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SwigluFwd.typedKernel

/-- Complete selected original FP32 source, including explicit input casts,
sigmoid, left-associated products, source masks and kernel ports. -/
def kernel (X Y OUT : RegionName) (sx sy so cols block : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, Y], outputs=[OUT]
    tl.region X = tl.float32, Y = tl.float32, OUT = tl.float32
    row = tl.program_id(0)
    start_col = tl.program_id(1) * $(block)
    X += row * $(sx)
    Y += row * $(sy)
    OUT += row * $(so)
    cols = start_col + tl.arange(0, $(block))
    x = tl.load(X + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < $(cols))
  }

/-- Independent value on the two original last-axis halves. -/
noncomputable def formula (x y : ℝ) : ℝ := x*y/(1+Real.exp (-x))

/-- Complete existing candidate: commute only the three integer row-stride
products, preserving FP32 loads, casts, sigmoid and multiplication order. -/
def indexCommutedKernel (X Y OUT : RegionName) (sx sy so cols block : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[X, Y], outputs=[OUT]
    tl.region X = tl.float32, Y = tl.float32, OUT = tl.float32
    row = tl.program_id(0)
    start_col = tl.program_id(1) * $(block)
    X += $(sx) * row
    Y += $(sy) * row
    OUT += $(so) * row
    cols = start_col + tl.arange(0, $(block))
    x = tl.load(X + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < $(cols), other=0.0).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < $(cols))
  }

end VeriTile.Bench.Repairs.SwigluFwd
