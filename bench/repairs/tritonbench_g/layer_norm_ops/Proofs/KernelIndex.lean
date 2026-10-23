import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelDispatch

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelIndex
open Admission
noncomputable section

theorem forward (c : ForwardCall) (checked : forwardCheck c=Bool.true)
    (row lane : Nat) (hr : row<c.common.rows) (hl : lane<blockSize c.common.columns) :
    row*c.common.columns+lane<2^31 := by
  have h := ((forwardCheck_iff c).mp checked).1
  have cap := h.2.2.2.2.2.2.2.2
  have hm := Nat.mul_le_mul_right c.common.columns (Nat.le_of_lt hr)
  exact lt_of_le_of_lt (Nat.add_le_add hm (Nat.le_of_lt hl)) cap

theorem backward (c : BackwardCall) (checked : backwardCheck c=Bool.true)
    (pid lane : Nat) (hp : pid<c.programs) (hl : lane<blockSize c.common.columns) :
    pid*c.rowWidth<2^31 ∧ (pid+1)*c.rowWidth<2^31 ∧
    pid*c.rowWidth*c.common.columns+lane<2^31 ∧ pid*c.common.columns+lane<2^31 := by
  obtain ⟨hc,_,_,_,_,_,_,_,_,_,hg,hpadded⟩ := (backwardCheck_iff c).mp checked
  have positive : 1≤c.common.columns := hc.1
  have he := Nat.mul_le_mul_right c.rowWidth (Nat.succ_le_of_lt hp)
  have hs := Nat.mul_le_mul_right c.rowWidth (Nat.le_of_lt hp)
  have scale : c.programs*c.rowWidth≤c.programs*c.rowWidth*c.common.columns := by
    simpa using Nat.mul_le_mul_left (c.programs*c.rowWidth) positive
  have bound : c.programs*c.rowWidth<2^31 :=
    lt_of_le_of_lt (scale.trans (Nat.le_add_right _ _)) hg
  refine ⟨lt_of_le_of_lt hs bound,lt_of_le_of_lt he bound,?_,?_⟩
  · exact lt_of_le_of_lt (Nat.add_le_add (Nat.mul_le_mul_right c.common.columns hs) (Nat.le_of_lt hl)) hg
  · exact lt_of_le_of_lt (Nat.add_le_add (Nat.mul_le_mul_right c.common.columns (Nat.le_of_lt hp))
      (Nat.le_of_lt hl)) hpadded

theorem saved_statistics (c : BackwardCall) (checked : backwardCheck c=Bool.true)
    (row : Nat) (hr : row<c.common.rows) :
    row*(KernelDispatch.backward c).meanStride<2^31 ∧ row*(KernelDispatch.backward c).rstdStride<2^31 := by
  have hc := ((backwardCheck_iff c).mp checked).1
  have scale : c.common.rows≤c.common.rows*c.common.columns := by
    simpa using Nat.mul_le_mul_left c.common.rows hc.1
  have cap : c.common.rows<2^31 := lt_of_le_of_lt (scale.trans (Nat.le_add_right _ _)) hc.2.2.2.2.2.2.2.2
  have bound : row<2^31 := lt_trans hr cap
  simp only [KernelDispatch.backward]
  cases c.common.rms.getD Bool.false <;> simp [bound]
  all_goals simpa using bound

#axiomsClean forward
#axiomsClean backward
#axiomsClean saved_statistics
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelIndex
