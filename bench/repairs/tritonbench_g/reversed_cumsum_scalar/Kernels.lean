import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete repaired source. Integer coordinates use the mathematical
projection of the explicit native signed-i64 calculations; NativeIndex proves
that projection for the actual backing and masks. Numerical operations retain
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

def source (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let o := h.output
  triton {
    tl.ports inputs=[s], outputs=[o]
    tl.region s = DTYPE, o = OUT_DTYPE
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv($(h.T), $(c.BT)) - $(1), -$(1), -$(1)) {
      start_t = i_t * $(c.BT)
      lanes = tl.arange(0, $(c.BT))
      active = lanes < $(h.T) - start_t
      times = start_t + tl.where(active, lanes, $(0))
      p_s = s + (i_bh // $(h.H)) * $(h.inputStrides.batch) + (i_bh % $(h.H)) * $(h.inputStrides.head) + times * $(h.inputStrides.time)
      p_o = o + (i_bh // $(h.H)) * $(h.outputStrides.batch) + (i_bh % $(h.H)) * $(h.outputStrides.head) + times * $(h.outputStrides.time)
      b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
      b_zz = tl.sum(b_s, axis=0)
      b_z += b_zz
      b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
      tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=active)
    }
  }

/-- The intended discrete candidate; every floating operation, cast, mask
and store remains in the same source order. -/
def indexCommuted (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let o := h.output
  triton {
    tl.ports inputs=[s], outputs=[o]
    tl.region s = DTYPE, o = OUT_DTYPE
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv($(h.T), $(c.BT)) - $(1), -$(1), -$(1)) {
      start_t = i_t * $(c.BT)
      lanes = tl.arange(0, $(c.BT))
      active = lanes < $(h.T) - start_t
      times = start_t + tl.where(active, lanes, $(0))
      p_s = s + $(h.inputStrides.batch) * (i_bh // $(h.H)) + $(h.inputStrides.head) * (i_bh % $(h.H)) + times * $(h.inputStrides.time)
      p_o = o + (i_bh // $(h.H)) * $(h.outputStrides.batch) + (i_bh % $(h.H)) * $(h.outputStrides.head) + times * $(h.outputStrides.time)
      b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
      b_zz = tl.sum(b_s, axis=0)
      b_z += b_zz
      b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
      tl.store(p_o, (b_o).to(p_o.dtype.element_ty), mask=active)
    }
  }

end Kernels
end VeriTile.Bench.Repairs.ReversedCumsumScalar
