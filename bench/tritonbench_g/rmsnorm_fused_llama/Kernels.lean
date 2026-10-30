import bench.tritonbench_g.rmsnorm_fused_llama.Sources.Mathematical_Kernels
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.IndexTransform
import VeriTile.Meta.StatementAudit

/-! Complete ordinary quotations for the literal/runtime original source and
its existing integer pointer-product candidate. Runtime epsilon is explicitly
declared without a dummy value or custom source frontend. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama
open VeriTile Triton FP.Structural

def kernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  row = tl.program_id(0)
  Y += row * $(N)
  X += row * $(N)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
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

def runtimeKernel (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps_arg : tl.float32
  row = tl.program_id(0)
  Y += row * $(N)
  X += row * $(N)
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

def indexCommutedKernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  row = tl.program_id(0)
  Y += $(N) * row
  X += row * $(N)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
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

def runtimeCandidate (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[X,W], outputs=[Y]
  tl.region X = tl.float32, W = tl.float32, Y = tl.float32
  tl.param eps_arg : tl.float32
  row = tl.program_id(0)
  Y += $(N) * row
  X += row * $(N)
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

theorem source_eq (X Y W : RegionName) (N B : Nat) (eps : ℝ) :
    kernel X Y W N B eps=
      VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.originalTyped .fp32 .fp32 .fp32 X Y W N N B eps := rfl

theorem candidate_eq (X Y W : RegionName) (N B : Nat) (eps : ℝ) :
    indexCommutedKernel X Y W N B eps=
      VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.typedKernel .fp32 .fp32 .fp32 Bool.true X Y W N N B eps := rfl
end VeriTile.Bench.Repairs.RmsnormFusedLlama
