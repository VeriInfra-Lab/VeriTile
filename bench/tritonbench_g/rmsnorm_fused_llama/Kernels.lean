import VeriTile.Triton
import VeriTile.Triton.Source
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Repaired two-pass Python source with runtime epsilon. The candidate changes
only the integer output-pointer product. Real calls bind the same runtime
argument before erasing precision; no separate literal implementation is kept. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama
noncomputable section
open VeriTile Triton FP.Structural

def runtimeKernel (X Y W : RegionName) (N B : Nat) (stride : Nat := N) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps_arg : tl.float32
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + eps_arg)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y.to(tl.float16)).to(tl.float32), mask=mask)
  }
  }

def runtimeCandidate (X Y W : RegionName) (N B : Nat) (stride : Nat := N) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps_arg : tl.float32
  row = tl.program_id(0)
  Y += $(stride) * row
  X += row * $(stride)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + eps_arg)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y.to(tl.float16)).to(tl.float32), mask=mask)
  }
  }

def kernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  (runtimeKernel X Y W N B).withScalarArguments [ScalarArgument.fp32 "eps_arg" eps]

def indexCommutedKernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  (runtimeCandidate X Y W N B).withScalarArguments [ScalarArgument.fp32 "eps_arg" eps]

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama
