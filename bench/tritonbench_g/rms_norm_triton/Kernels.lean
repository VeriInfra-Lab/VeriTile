/-
Source-audit target: rms_norm_kernel
Source-audit Lean: kernel
Selected FP32 source quotation. Native instruction/compiler binding is explicit
in FPEquiv; this lexical metadata does not certify a compiler profile.
-/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton

/-- Original kernel, including the input cast, runtime-epsilon ABI cast and
output cast before weighting. Row/column strides are source parameters. -/
def kernel (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region Y = tl.float32, X = tl.float32, W = tl.float32
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(tl.float32) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton
