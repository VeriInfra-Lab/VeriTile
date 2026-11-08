import VeriTile.Triton
import VeriTile.Triton.Source
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Complete two-pass Python bodies with runtime epsilon. The existing
candidate changes only the integer output-pointer product. Both casts and
the implicit destination conversion remain part of the actual source. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama
noncomputable section
open VeriTile Triton FP.Structural

def callSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("W",.fp32)]
  scalars := [("stride",.integer),("N",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W"]
  writePorts := ["Y"]

def runtimeKernel (X Y W : RegionName) (N B : Nat) (stride : Nat := N) : ComputeKernel :=
  let BLOCK_SIZE := B
  with_triton_call callSignature call_args [stride := stride,N := N] in triton {
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

def runtimeCandidateKernel (X Y W : RegionName) (N B : Nat) (stride : Nat := N) : ComputeKernel :=
  let BLOCK_SIZE := B
  with_triton_call callSignature call_args [stride := stride,N := N] in triton {
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

def kernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  (runtimeKernel X Y W N B).withScalarArguments [ScalarArgument.fp32 "eps" eps]

def indexCommutedKernel (X Y W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  (runtimeCandidateKernel X Y W N B).withScalarArguments [ScalarArgument.fp32 "eps" eps]

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama
