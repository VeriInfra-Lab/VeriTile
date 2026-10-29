import VeriTile.Triton.DSL
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Kernels

/-! Shared complete repaired source, retaining triangular FP32 dot, explicit
load/output conversions and the post-store carry update. Host preparation is
a guarded contiguous FP32 Tensor. All coordinates remain relative to its actual data pointer.
Native i64 casts are projected to mathematical coordinates here and checked
separately against actual allocation bounds. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum
open VeriTile.Triton

abbrev Config := ReversedCumsumScalar.Config

def configurations : List Config := [⟨16,2⟩,⟨16,4⟩,⟨16,8⟩,⟨32,2⟩,⟨32,4⟩,⟨32,8⟩,⟨64,2⟩,⟨64,4⟩,⟨64,8⟩]

def BS : Nat := 32

structure Call where
  B : Nat
  H : Nat
  T : Nat
  S : Nat
  input : RegionName
  output : RegionName
  inputDType : FloatDType := .fp32
  outputDType : FloatDType := .fp32

/-- Actual contiguous prepared storage, with time and feature coordinates
flattened only in the generic physical-storage adapter. -/
def Call.flat (h : Call) : ReversedCumsumScalar.Call where
  B := h.B
  H := h.H
  T := h.T*h.S
  input := h.input
  output := h.output
  inputStrides := ⟨h.H*(h.T*h.S),h.T*h.S,1⟩
  outputStrides := ⟨h.H*(h.T*h.S),h.T*h.S,1⟩
  inputDType := h.inputDType
  outputDType := h.outputDType

def Call.timeView (h : Call) : ReversedCumsumScalar.Call := {h.flat with T:=h.T}
def Call.featureView (h : Call) : ReversedCumsumScalar.Call := {h.flat with T:=h.S}

namespace Kernels
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def source (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  triton {
    tl.ports inputs=[s], outputs=[z]
    tl.region s = DTYPE, z = OUT_DTYPE
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1)
    o_i = tl.arange(0, $(c.BT))
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([$(BS)], dtype=tl.float32)
    for i_t in range(tl.cdiv($(h.T), $(c.BT)) - $(1), -$(1), -$(1)) {
      start_t = i_t * $(c.BT)
      start_s = i_s * $(BS)
      lane_t = tl.arange(0, $(c.BT))
      lane_s = tl.arange(0, $(BS))
      active_t = lane_t < $(h.T) - start_t
      active_s = lane_s < $(h.S) - start_s
      times = start_t + tl.where(active_t, lane_t, $(0))
      features = start_s + tl.where(active_s, lane_s, $(0))
      active = active_t[:, None] & active_s[None, :]
      p_s = s + i_bh * $(h.T*h.S) + times[:, None] * $(h.S) + features[None, :]
      p_z = z + i_bh * $(h.T*h.S) + times[:, None] * $(h.S) + features[None, :]
      b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
      b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
      tl.store(p_z, (b_c).to(p_z.dtype.element_ty), mask=active)
      if i_t >= $(0) {
        b_z += tl.sum(b_s, 0)
      }
    }
  }

/-- The intended discrete candidate; every floating operation, cast, mask
and store remains in the same source order. -/
def compareRight (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  triton {
    tl.ports inputs=[s], outputs=[z]
    tl.region s = DTYPE, z = OUT_DTYPE
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1)
    o_i = tl.arange(0, $(c.BT))
    m_s = tl.where(o_i[None, :] >= o_i[:, None], 1.0, 0.0)
    b_z = tl.zeros([$(BS)], dtype=tl.float32)
    for i_t in range(tl.cdiv($(h.T), $(c.BT)) - $(1), -$(1), -$(1)) {
      start_t = i_t * $(c.BT)
      start_s = i_s * $(BS)
      lane_t = tl.arange(0, $(c.BT))
      lane_s = tl.arange(0, $(BS))
      active_t = lane_t < $(h.T) - start_t
      active_s = lane_s < $(h.S) - start_s
      times = start_t + tl.where(active_t, lane_t, $(0))
      features = start_s + tl.where(active_s, lane_s, $(0))
      active = active_t[:, None] & active_s[None, :]
      p_s = s + i_bh * $(h.T*h.S) + times[:, None] * $(h.S) + features[None, :]
      p_z = z + i_bh * $(h.T*h.S) + times[:, None] * $(h.S) + features[None, :]
      b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
      b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
      tl.store(p_z, (b_c).to(p_z.dtype.element_ty), mask=active)
      if i_t >= $(0) {
        b_z += tl.sum(b_s, 0)
      }
    }
  }

end Kernels
end VeriTile.Bench.Repairs.ReversedCumsum
