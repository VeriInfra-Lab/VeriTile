import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Kernels

/-! Shared complete repaired source, retaining triangular FP32 dot, explicit
load/output conversions and the post-store carry update. Host preparation is
a guarded contiguous FP32 Tensor. All coordinates remain relative to its actual data pointer.
Native i64 casts are visible in the raw body; the actual allocation bounds
and storage refinements retain their existing separate proofs. -/
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

def callSignature : CallSignature where
  pointers := [("s",.inputFloat), ("z",.outputFloat)]
  scalars := []
  constexpr := ["s_s_h", "s_s_t", "s_s_d", "T", "S", "BT", "BS"]

def source (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  let s_s_h := h.T*h.S
  let s_s_t := h.S
  let s_s_d := (1 : Nat)
  let T := h.T
  let S := h.S
  let BT := c.BT
  let BS := (BS : Nat)
  with_triton_signature callSignature in triton {
    i_s, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        start_s = i_s * BS
        lane_t = tl.arange(0, BT).to(tl.int64)
        lane_s = tl.arange(0, BS).to(tl.int64)
        active_t = lane_t < T - start_t
        active_s = lane_s < S - start_s
        times = start_t + tl.where(active_t, lane_t, 0)
        features = start_s + tl.where(active_s, lane_s, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        p_z = z + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        # [BT, BS]
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)
  }

def compareRight (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  let s_s_h := h.T*h.S
  let s_s_t := h.S
  let s_s_d := (1 : Nat)
  let T := h.T
  let S := h.S
  let BT := c.BT
  let BS := (BS : Nat)
  with_triton_signature callSignature in triton {
    i_s, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[None, :] >= o_i[:, None], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        start_s = i_s * BS
        lane_t = tl.arange(0, BT).to(tl.int64)
        lane_s = tl.arange(0, BS).to(tl.int64)
        active_t = lane_t < T - start_t
        active_s = lane_s < S - start_s
        times = start_t + tl.where(active_t, lane_t, 0)
        features = start_s + tl.where(active_s, lane_s, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        p_z = z + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        # [BT, BS]
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)
  }

end Kernels
end VeriTile.Bench.Repairs.ReversedCumsum
