import bench.tritonbench_g.attn_fwd_triton.Proofs.NativeArithmetic
/-! Ordinary selected-backend launch limits and their integer consequences,
shared by the independent source interpretations. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.NativeLaunch
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section
structure Shape where
  batches : Nat
  heads : Nat
  context : Nat
structure CudaLaunchable (p : Shape) (BD : Nat) : Prop where
  gridX : (p.context+127)/128 ≤ 2147483647
  gridY : p.batches*p.heads ≤ 65535
  arange : BD ≤ 1048576

theorem context (p : Shape) (BD : Nat) (h : CudaLaunchable p BD) :
    p.context ≤ 274877906816 := by
  have := h.gridX
  omega

theorem cdiv_inputs (p : Shape) (BD : Nat) (h : CudaLaunchable p BD) :
    signed64 p.context=(p.context : Int) ∧
      signed64 (p.context+127)=(p.context+127 : Nat) ∧
      signed64 (p.context+63)=(p.context+63 : Nat) := by
  have := context p BD h
  exact ⟨signed64_exact (by omega),signed64_exact (by omega),signed64_exact (by omega)⟩

theorem row_arithmetic (p : Shape) (BD start row : Nat) (h : CudaLaunchable p BD)
    (hs : start*128<p.context) (hr : row<128) :
    signed64 (start*128+row)=(start*128+row : Nat) ∧
      signed64 ((start+1)*128)=((start+1)*128 : Nat) ∧
      signed64 (min ((start+1)*128) p.context)=(min ((start+1)*128) p.context : Nat) := by
  have := context p BD h
  exact ⟨signed64_exact (by omega),signed64_exact (by omega),signed64_exact (by omega)⟩

theorem key_arithmetic (p : Shape) (BD cursor lane : Nat) (h : CudaLaunchable p BD)
    (hc : cursor<p.context) (hl : lane<64) :
    signed64 (p.context-cursor)=(p.context : Int)-(cursor : Int) ∧
      signed64 (cursor+lane)=(cursor+lane : Nat) ∧
      signed64 (cursor+64)=(cursor+64 : Nat) := by
  have := context p BD h
  refine ⟨?_,signed64_exact (by omega),signed64_exact (by omega)⟩
  rw [signed64_exact (by omega)]
  exact Nat.cast_sub (Nat.le_of_lt hc)

theorem feature_lanes (p : Shape) (BD lane : Nat) (h : CudaLaunchable p BD) (hl : lane<BD) :
    signed64 lane=(lane : Int) := by
  have := h.arange
  exact signed64_exact (by omega)

theorem packed_offsets (p : Shape) (BD head : Nat) (h : CudaLaunchable p BD)
    (hh : head<p.batches*p.heads) :
    signed64 (head*((p.context+127)/128))=(head*((p.context+127)/128) : Nat) ∧
      signed64 (head*((p.context+63)/64))=(head*((p.context+63)/64) : Nat) := by
  have hctx := context p BD h
  have hhead : head≤65535 := le_trans (Nat.le_of_lt hh) h.gridY
  have hq : (p.context+127)/128≤2147483647 := h.gridX
  have hk : (p.context+63)/64≤4294967294 := by omega
  have hqm := Nat.mul_le_mul hhead hq
  have hkm := Nat.mul_le_mul hhead hk
  exact ⟨signed64_exact (by omega),signed64_exact (by omega)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.NativeLaunch
