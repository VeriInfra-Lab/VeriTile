import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! The repaired source keeps the original nine configurations, FP32 load
promotion, lower triangular dot with TF32 disabled, store and carry update.
Masked inactive loads use explicit zero. The raw Python body retains every
signed-64 cast. NativeIndex retains the actual storage and stride refinement obligations. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector
open VeriTile.Triton

structure Strides where
  batch : Nat
  head : Nat
  time : Nat
  channel : Nat

structure Call where
  B : Nat
  H : Nat
  T : Nat
  S : Nat
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

def BS : Nat := 32

def configurations : List Config :=
  [⟨16,2⟩,⟨16,4⟩,⟨16,8⟩,⟨32,2⟩,⟨32,4⟩,⟨32,8⟩,⟨64,2⟩,⟨64,4⟩,⟨64,8⟩]

def Strides.base (strides : Strides) (H row : Nat) : Nat :=
  row / H * strides.batch + row % H * strides.head

def Strides.address (strides : Strides) (H row time channel : Nat) : Nat :=
  strides.base H row + time * strides.time + channel * strides.channel

namespace Kernels
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def callSignature : CallSignature where
  pointers := [("s",.inputFloat),("z",.outputFloat)]
  scalars := []
  constexpr := ["H", "s_s_b", "s_s_h", "s_s_t", "s_s_d", "s_z_b", "s_z_h", "s_z_t", "s_z_d", "T", "S", "BT", "BS"]

def source (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  let H := h.H
  let s_s_b := h.inputStrides.batch
  let s_s_h := h.inputStrides.head
  let s_s_t := h.inputStrides.time
  let s_s_d := h.inputStrides.channel
  let s_z_b := h.outputStrides.batch
  let s_z_h := h.outputStrides.head
  let s_z_t := h.outputStrides.time
  let s_z_d := h.outputStrides.channel
  let T := h.T
  let S := h.S
  let BT := c.BT
  let BS := (BS : Nat)
  with_triton_signature callSignature in triton {
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    b_z = tl.zeros([BS], dtype=tl.float32)
    start_s = i_s * BS
    lanes_s = tl.arange(0, BS).to(tl.int64)
    active_s = lanes_s < S - start_s
    columns = start_s + tl.where(active_s, lanes_s, 0)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64)):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes_t = tl.arange(0, BT).to(tl.int64)
        active_t = lanes_t < T - start_t
        times = start_t + tl.where(active_t, lanes_t, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_b * s_s_b + i_h * s_s_h + times[:, None] * s_s_t + columns[None, :] * s_s_d
        p_z = z + i_b * s_z_b + i_h * s_z_h + times[:, None] * s_z_t + columns[None, :] * s_z_d
        # [BT, BS]; zero inactive lanes before the triangular dot.
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)
  }

def sumLeft (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  let H := h.H
  let s_s_b := h.inputStrides.batch
  let s_s_h := h.inputStrides.head
  let s_s_t := h.inputStrides.time
  let s_s_d := h.inputStrides.channel
  let s_z_b := h.outputStrides.batch
  let s_z_h := h.outputStrides.head
  let s_z_t := h.outputStrides.time
  let s_z_d := h.outputStrides.channel
  let T := h.T
  let S := h.S
  let BT := c.BT
  let BS := (BS : Nat)
  with_triton_signature callSignature in triton {
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    b_z = tl.zeros([BS], dtype=tl.float32)
    start_s = i_s * BS
    lanes_s = tl.arange(0, BS).to(tl.int64)
    active_s = lanes_s < S - start_s
    columns = start_s + tl.where(active_s, lanes_s, 0)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64)):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes_t = tl.arange(0, BT).to(tl.int64)
        active_t = lanes_t < T - start_t
        times = start_t + tl.where(active_t, lanes_t, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_b * s_s_b + i_h * s_s_h + times[:, None] * s_s_t + columns[None, :] * s_s_d
        p_z = z + i_b * s_z_b + i_h * s_z_h + times[:, None] * s_z_t + columns[None, :] * s_z_d
        # [BT, BS]; zero inactive lanes before the triangular dot.
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)
        if i_t >= 0:
            b_z = tl.sum(b_s, 0) + b_z
  }

end Kernels
end VeriTile.Bench.Repairs.ChunkCumsumVector
