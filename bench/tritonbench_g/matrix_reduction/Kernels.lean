import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Erasure

/-! Exact original block-pointer and existing affine-address Python bodies.
The wrapper selects FP16 or FP32 storage; dtype and all runtime stride call
bindings live outside the quotations. The unused stride_y is retained in the
signature. No source cast, address or statement is added to either body. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
open VeriTile.Triton

inductive Storage where
  | fp16 | fp32
  deriving DecidableEq, Repr

def signature16 : CallSignature where
  pointers := [("x_ptr",.fp16),("y_ptr",.fp16)]
  scalars := [("stride_xm",.integer),("stride_xn",.integer),("stride_y",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]
  readPorts := ["x_ptr"]
  writePorts := ["y_ptr"]

def source16 (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call signature16 call_args
    [stride_xm := stride_xm, stride_xn := stride_xn, stride_y := stride_y] in
  triton {
    x_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)
  }

def candidate16 (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call signature16 call_args
    [stride_xm := stride_xm, stride_xn := stride_xn, stride_y := stride_y] in
  triton {
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptr + offsets)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)
  }

def signature32 : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32)]
  scalars := [("stride_xm",.integer),("stride_xn",.integer),("stride_y",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]
  readPorts := ["x_ptr"]
  writePorts := ["y_ptr"]

def source32 (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call signature32 call_args
    [stride_xm := stride_xm, stride_xn := stride_xn, stride_y := stride_y] in
  triton {
    x_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)
  }

def candidate32 (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call signature32 call_args
    [stride_xm := stride_xm, stride_xn := stride_xn, stride_y := stride_y] in
  triton {
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptr + offsets)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)
  }

def kernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => source16
  | .fp32 => source32

def candidateKernel (storage : Storage) : RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel :=
  match storage with
  | .fp16 => candidate16
  | .fp32 => candidate32

#guard_msgs (drop info) in
#check kernel.eq_1
#guard_msgs (drop info) in
#check kernel.eq_2
end VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton

/-- Compatibility mathematical view of the full actual FP32 source. Precision
is erased solely for the independent Real proofs. It is not an FP claim. -/
def load_reduce_kernel (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  (TypedSource.source32 x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M BLOCK_N).eraseDType

end VeriTile.Bench.TritonBenchG.MatrixReduction
