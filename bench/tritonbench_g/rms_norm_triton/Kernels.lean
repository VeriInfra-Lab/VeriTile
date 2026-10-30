import bench.tritonbench_g.rms_norm_triton.Sources.Mathematical_Kernels

/-! The checked repair's JIT AST is identical to the original FP32 source.
The original reuses the public readable shared DSL; the complete existing
index-commuted candidate is quoted below. Only the Python guards differ.
`check_contract.py` checks the exact JIT AST and guard expressions. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Triton

def kernel (Y X W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.RmsNormTriton.kernel Y X W N 1 N 1 N eps B

theorem source_identity (Y X W : RegionName) (N B : Nat) (eps : ℝ) :
    kernel Y X W N B eps =
      VeriTile.Bench.TritonBenchG.RmsNormTriton.kernel Y X W N 1 N 1 N eps B := rfl

/-- Reuse the already specified source candidate: commute only the two
natural row-base products. The FP32 casts and floating expression are intact. -/
def indexCommutedKernel (Y X W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
  tl.ports inputs=[X, W], outputs=[Y]
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32
  pid = tl.program_id(0)
  Y += $(N) * pid
  X += $(N) * pid
  mask = tl.arange(0, $(B)) < $(N)
  cols = tl.arange(0, $(B))
  x = tl.load(X + cols * $(1), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  w = tl.load(W + tl.arange(0, $(B)), mask=mask, other=0.0)
  y = (x * rrms).to(tl.float32) * w
  tl.store(Y + cols * $(1), y, mask=mask)
}

end VeriTile.Bench.Repairs.RmsNormTriton
