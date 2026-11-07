import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

/-! The two canonical quotations copy the original and existing candidate
Python bodies. Runtime epsilon remains an FP32 argument. Compatibility call
helpers bind it before execution and introduce no cast inside the body. -/
namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("Y",.fp32),("X",.fp32),("W",.fp32)]
  scalars := [("y_stride_r",.i32),("y_stride_c",.i32),("x_stride_r",.i32),("x_stride_c",.i32),("N",.i32),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["X","W"]
  writePorts := ["Y"]

def rawKernel (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [y_stride_r := y_stride_r,
    y_stride_c := y_stride_c, x_stride_r := x_stride_r, x_stride_c := x_stride_c,
    N := N] in triton {
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
  }

def rawCommutedKernel (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call callSignature call_args [y_stride_r := y_stride_r,
    y_stride_c := y_stride_c, x_stride_r := x_stride_r, x_stride_c := x_stride_c,
    N := N] in triton {
    pid = tl.program_id(0)
    Y += y_stride_r * pid
    X += x_stride_r * pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)
  }

/-- Convert the Real-valued compatibility argument at the call boundary. -/
noncomputable def epsilonArgument (eps : ℝ) : ScalarArgument :=
  ⟨"eps",.real,.compute (.alg .fp32 (.castFloat .real .real (.const eps)))⟩

noncomputable def kernel (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  (rawKernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE).withScalarArguments
    [epsilonArgument eps]

end VeriTile.Bench.TritonBenchG.RmsNormTriton

namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsNormTriton

def runtimeKernel (Y X W : RegionName) (N B : Nat) : ComputeKernel :=
  rawKernel Y X W N 1 N 1 N B

def runtimeCandidate (Y X W : RegionName) (N B : Nat) : ComputeKernel :=
  rawCommutedKernel Y X W N 1 N 1 N B

noncomputable def kernel (Y X W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  (runtimeKernel Y X W N B).withScalarArguments [epsilonArgument eps]

noncomputable def indexCommutedKernel (Y X W : RegionName) (N B : Nat) (eps : ℝ) : ComputeKernel :=
  (runtimeCandidate Y X W N B).withScalarArguments [epsilonArgument eps]

end VeriTile.Bench.Repairs.RmsNormTriton
