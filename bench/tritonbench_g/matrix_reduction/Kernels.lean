import VeriTile.Triton
import VeriTile.Triton.DSL.PythonCallSyntax

/-! Complete original block-pointer and existing affine-address Python bodies.
Caller dtypes, runtime strides and both constexpr extents live outside. FP16
and FP32 are both retained; the Real API is an erasure of this shared source. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
open VeriTile.Triton
inductive Storage where
  | fp16 | fp32
  deriving DecidableEq, Repr

def callSignature (storage : Storage) : CallSignature where
  pointers := [("x_ptr",match storage with | .fp16 => .fp16 | .fp32 => .fp32),("y_ptr",match storage with | .fp16 => .fp16 | .fp32 => .fp32)]
  scalars := [("stride_xm",.integer),("stride_xn",.integer),("stride_y",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]
  readPorts := ["x_ptr"]
  writePorts := ["y_ptr"]

def originalForKernel (storage : Storage) (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature storage) call_args
    [stride_xm := stride_xm, stride_xn := stride_xn, stride_y := stride_y] in triton {
    x_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)
  }

def candidateForKernel (storage : Storage) (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature storage) call_args
    [stride_xm := stride_xm, stride_xn := stride_xn, stride_y := stride_y] in triton {
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptr + offsets)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)
  }
def kernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => originalForKernel .fp16
  | .fp32 => originalForKernel .fp32

def candidateKernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => candidateForKernel .fp16
  | .fp32 => candidateForKernel .fp32

#guard_msgs (drop info) in
#check kernel.eq_1
#guard_msgs (drop info) in
#check kernel.eq_2

end VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton
/-- Real mathematical view of the actual FP32 source, without an additional
implementation. All typed storage profiles are connected by source_shape. -/
def load_reduce_kernel (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  (TypedSource.originalForKernel .fp32 x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M BLOCK_N).eraseDType
end VeriTile.Bench.TritonBenchG.MatrixReduction
