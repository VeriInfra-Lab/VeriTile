import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.rmsnorm_fused_llama.Kernels
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.IndexTransform
import VeriTile.Meta.StatementAudit

/-! Complete ordinary quotations for the literal/runtime original source and
its existing integer pointer-product candidate. Runtime epsilon is explicitly
declared without a dummy value or custom source frontend. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama
open VeriTile Triton FP.Structural

def callSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("W",.fp32)]
  scalars := [("stride",.integer),("N",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W"]
  writePorts := ["Y"]


def kernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [stride := N, N := N, eps := eps] in
  triton {
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)
  }

def runtimeKernel (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [stride := N, N := N] in
  triton {
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)
  }

def indexCommutedKernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [stride := N, N := N, eps := eps] in
  triton {
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += stride * row
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)
  }

def runtimeCandidate (X Y W : RegionName) (N B : Nat) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [stride := N, N := N] in
  triton {
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += stride * row
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)
  }

theorem source_eq (X Y W : RegionName) (N B : Nat) (eps : ℝ) :
    kernel X Y W N B eps=
      VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.originalTyped .fp32 .fp32 .fp32 X Y W N N B eps := rfl

theorem candidate_eq (X Y W : RegionName) (N B : Nat) (eps : ℝ) :
    indexCommutedKernel X Y W N B eps=
      VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.typedKernel .fp32 .fp32 .fp32 Bool.true X Y W N N B eps := rfl
end VeriTile.Bench.Repairs.RmsnormFusedLlama
