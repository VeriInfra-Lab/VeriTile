import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! The repaired source keeps the original nine configurations, FP32 load
promotion, lower triangular dot with TF32 disabled, store and carry update.
Masked inactive loads now use explicit zero. NativeIndex connects the repaired
natural coordinates to the source's actual signed-64 arithmetic. -/
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
    i_b = i_bh // $(h.H)
    i_h = i_bh % $(h.H)
    o_i = tl.arange(0, $(c.BT))
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([$(BS)], dtype=tl.float32)
    start_s = i_s * $(BS)
    lanes_s = tl.arange(0, $(BS))
    active_s = lanes_s < $(h.S) - start_s
    columns = start_s + tl.where(active_s, lanes_s, $(0))
    for i_t in range($(0), tl.cdiv($(h.T), $(c.BT)), $(1)) {
      start_t = i_t * $(c.BT)
      lanes_t = tl.arange(0, $(c.BT))
      active_t = lanes_t < $(h.T) - start_t
      times = start_t + tl.where(active_t, lanes_t, $(0))
      active = active_t[:, None] & active_s[None, :]
      p_s = s + i_b * $(h.inputStrides.batch) + i_h * $(h.inputStrides.head) + times[:, None] * $(h.inputStrides.time) + columns[None, :] * $(h.inputStrides.channel)
      p_z = z + i_b * $(h.outputStrides.batch) + i_h * $(h.outputStrides.head) + times[:, None] * $(h.outputStrides.time) + columns[None, :] * $(h.outputStrides.channel)
      b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
      b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
      tl.store(p_z, (b_c).to(p_z.dtype.element_ty), mask=active)
      if i_t >= $(0) { b_z += tl.sum(b_s, 0) }
    }
  }

/-- The intended candidate changes only the FP32 carry-add operand order. -/
def sumLeft (h : Call) (c : Config) : ComputeKernel :=
  let dtype := h.inputDType
  let out_dtype := h.outputDType
  let s := h.input
  let z := h.output
  triton {
    tl.ports inputs=[s], outputs=[z]
    tl.region s = DTYPE, z = OUT_DTYPE
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // $(h.H)
    i_h = i_bh % $(h.H)
    o_i = tl.arange(0, $(c.BT))
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([$(BS)], dtype=tl.float32)
    start_s = i_s * $(BS)
    lanes_s = tl.arange(0, $(BS))
    active_s = lanes_s < $(h.S) - start_s
    columns = start_s + tl.where(active_s, lanes_s, $(0))
    for i_t in range($(0), tl.cdiv($(h.T), $(c.BT)), $(1)) {
      start_t = i_t * $(c.BT)
      lanes_t = tl.arange(0, $(c.BT))
      active_t = lanes_t < $(h.T) - start_t
      times = start_t + tl.where(active_t, lanes_t, $(0))
      active = active_t[:, None] & active_s[None, :]
      p_s = s + i_b * $(h.inputStrides.batch) + i_h * $(h.inputStrides.head) + times[:, None] * $(h.inputStrides.time) + columns[None, :] * $(h.inputStrides.channel)
      p_z = z + i_b * $(h.outputStrides.batch) + i_h * $(h.outputStrides.head) + times[:, None] * $(h.outputStrides.time) + columns[None, :] * $(h.outputStrides.channel)
      b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
      b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
      tl.store(p_z, (b_c).to(p_z.dtype.element_ty), mask=active)
      if i_t >= $(0) { b_z = tl.sum(b_s, 0) + b_z }
    }
  }

end Kernels
end VeriTile.Bench.Repairs.ChunkCumsumVector
