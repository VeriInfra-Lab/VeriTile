import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Complete repaired source. All explicit signed-i64 casts remain in the original raw body. NativeIndex
retains the actual backing and mask refinement obligations. Numerical operations retain
source FP32 precision, forward scan, reduction, carry and subtraction/addition
order. Both independent interpretations import this one source. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar
open VeriTile.Triton

structure Strides where
  batch : Nat
  head : Nat
  time : Nat

structure Call where
  B : Nat
  H : Nat
  T : Nat
  input : RegionName
  output : RegionName
  inputStrides : Strides
  outputStrides : Strides
  inputDType : FloatDType := .fp32
  outputDType : FloatDType := .fp32

structure Config where
  BT : Nat
  warps : Nat
  deriving DecidableEq

def configurations : List Config := [⟨16,2⟩,⟨32,4⟩,⟨32,2⟩,⟨64,8⟩,⟨64,4⟩]

def Strides.base (strides : Strides) (H row : Nat) : Nat :=
  row / H * strides.batch + row % H * strides.head

def Strides.address (strides : Strides) (H row time : Nat) : Nat :=
  strides.base H row + time * strides.time

namespace Kernels
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def callSignature : CallSignature where
  pointers := [("s",.inputFloat), ("o",.outputFloat)]
  scalars := []
  constexpr := ["T", "H", "stride_sb", "stride_sh", "stride_st", "stride_ob", "stride_oh", "stride_ot", "BT"]

def source (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let o := h.output
  let T := h.T
  let H := h.H
  let stride_sb := h.inputStrides.batch
  let stride_sh := h.inputStrides.head
  let stride_st := h.inputStrides.time
  let stride_ob := h.outputStrides.batch
  let stride_oh := h.outputStrides.head
  let stride_ot := h.outputStrides.time
  let BT := c.BT
  with_triton_signature callSignature in triton {
    i_bh = tl.program_id(0).to(tl.int64)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes = tl.arange(0, BT).to(tl.int64)
        active = lanes < T - start_t
        times = start_t + tl.where(active, lanes, 0)
        p_s = s + (i_bh // H) * stride_sb + (i_bh % H) * stride_sh + times * stride_st
        p_o = o + (i_bh // H) * stride_ob + (i_bh % H) * stride_oh + times * stride_ot
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=active)
  }

def indexCommuted (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let o := h.output
  let T := h.T
  let H := h.H
  let stride_sb := h.inputStrides.batch
  let stride_sh := h.inputStrides.head
  let stride_st := h.inputStrides.time
  let stride_ob := h.outputStrides.batch
  let stride_oh := h.outputStrides.head
  let stride_ot := h.outputStrides.time
  let BT := c.BT
  with_triton_signature callSignature in triton {
    i_bh = tl.program_id(0).to(tl.int64)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes = tl.arange(0, BT).to(tl.int64)
        active = lanes < T - start_t
        times = start_t + tl.where(active, lanes, 0)
        p_s = s + stride_sb * (i_bh // H) + stride_sh * (i_bh % H) + times * stride_st
        p_o = o + (i_bh // H) * stride_ob + (i_bh % H) * stride_oh + times * stride_ot
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=active)
  }

end Kernels
end VeriTile.Bench.Repairs.ReversedCumsumScalar
