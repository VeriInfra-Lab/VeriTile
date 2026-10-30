import bench.tritonbench_g.fast_rms_layernorm.Kernels
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Settings

def block (cols : Nat) : Nat := 2^(Nat.clog 2 cols)
def warps (cols : Nat) : Nat :=
  if block cols≥32768 then 32 else if block cols≥8192 then 16 else if block cols≥2048 then 8 else 4

theorem covered (cols : Nat) : cols≤block cols :=
  (Nat.clog_le_iff_le_pow (by decide : 1<2)).mp (Nat.le_refl _)

def calculate (rows cols : Nat) (gemma : Bool) (eps : ℝ) : Except String Call :=
  if positive : 0<cols then
    if limit : block cols≤65536 then
      .ok ⟨rows,cols,block cols,positive,covered cols,limit,gemma,eps⟩
    else .error "source fused-block guard"
  else .error "nonempty final dimension required"

theorem rejected_zero (rows : Nat) (gemma : Bool) (eps : ℝ) :
    calculate rows 0 gemma eps=.error "nonempty final dimension required" := by simp [calculate]

theorem rejected_block (rows cols : Nat) (gemma : Bool) (eps : ℝ)
    (positive : 0<cols) (unsupported : 65536<block cols) :
    calculate rows cols gemma eps=.error "source fused-block guard" := by
  simp [calculate,positive,show ¬block cols≤65536 from Nat.not_le.mpr unsupported]

theorem accepted (rows cols : Nat) (gemma : Bool) (eps : ℝ)
    (positive : 0<cols) (supported : block cols≤65536) :
    ∃ h,calculate rows cols gemma eps=.ok h ∧ h.rows=rows ∧ h.cols=cols ∧ h.block=block cols ∧ h.gemma=gemma ∧ h.eps=eps := by
  exact ⟨⟨rows,cols,block cols,positive,covered cols,supported,gemma,eps⟩,by simp [calculate,positive,supported],rfl,rfl,rfl,rfl,rfl⟩

#axiomsClean accepted
end VeriTile.Bench.Repairs.FastRmsLayernorm.Settings
