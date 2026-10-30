import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.quantize_kv_transform.IO
namespace VeriTile.Bench.Repairs.QuantizeKvTransform.LogicalReader
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
def formula (p : Parameters) (count : Nat) (xs : LogicalView.Inputs p count) : LogicalView.Outputs p count where
  code := fun t h c =>
    let x := LogicalView.row xs t h
    if mathematicalMaximum x p.columns=0 then 0 else
      let z := xs.values t h c/(mathematicalMaximum x p.columns/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt
  scale := fun t h => mathematicalMaximum (LogicalView.row xs t h) p.columns/127

/-- The complete prior judgment is equivalent after bijective tensor reindexing. -/
theorem judgment_iff (rest : Algebra ℝ) (p : Parameters) (count : Nat) (order : List Nat) :
    (LogicalView.quantizeIO rest p count order).ImplementsOn LogicalView.Destinations (formula p count) ↔
    (ReaderBefore.quantizeIO rest p count order).ImplementsOn (IO.Destinations p count) (ReaderBefore.mathematical p count) := by
  unfold LogicalProgramIO.ImplementsOn LogicalProgramIO.lift
  apply Iff.of_eq
  congr 1
  · funext vals
    apply propext
    simp only [LogicalView.quantizeIO,LogicalView.Destinations,LogicalView.inputEquiv,Equiv.coe_fn_mk,IO.Destinations]
    constructor
    · rintro ⟨nonnegative,unique⟩
      refine ⟨nonnegative,?_⟩
      intro a b ha hb eq
      exact congrArg Fin.val (unique ⟨a,ha⟩ ⟨b,hb⟩ (by simpa [IO.destination,ha,hb] using eq))
    · rintro ⟨nonnegative,unique⟩
      refine ⟨nonnegative,?_⟩
      intro a b eq
      apply Fin.ext
      exact unique a.val b.val a.isLt b.isLt (by simpa [IO.destination,a.isLt,b.isLt] using eq)
  · funext vals o j
    change IO.Inputs ℝ p count at vals
    have row_eq (t : Fin count) (h : Fin p.heads) :
        LogicalView.row (LogicalView.inputEquiv p count vals) t h = IO.values vals t.val h.val := by
      funext c
      change (if hc : c<p.columns then vals (0 : Fin 2) (t,h,⟨c,hc⟩,()) else (0 : ℝ)) = _
      by_cases hc : c<p.columns
      · have valid : t.val<count ∧ h.val<p.heads ∧ c<p.columns := ⟨t.isLt,h.isLt,hc⟩
        simp only [IO.values,dif_pos valid,dif_pos hc]
        rfl
      · have invalid : ¬(t.val<count ∧ h.val<p.heads ∧ c<p.columns) := fun ht => hc ht.2.2
        simp only [IO.values,dif_neg invalid,dif_neg hc]
        rfl
    fin_cases o
    · change (formula p count (LogicalView.inputEquiv p count vals)).code j.1 j.2.1 j.2.2.1 = _
      dsimp only [formula]
      rw [row_eq]
      have element_eq : (LogicalView.inputEquiv p count vals).values j.1 j.2.1 j.2.2.1 =
          IO.values vals j.1.val j.2.1.val j.2.2.1.val := by
        change vals (0 : Fin 2) (j.1,j.2.1,j.2.2.1,()) = _
        have valid : j.1.val<count ∧ j.2.1.val<p.heads ∧ j.2.2.1.val<p.columns := ⟨j.1.isLt,j.2.1.isLt,j.2.2.1.isLt⟩
        simp only [IO.values,dif_pos valid]
        rfl
      rw [element_eq]
      rfl
    · change mathematicalMaximum (LogicalView.row (LogicalView.inputEquiv p count vals) j.1 j.2.1) p.columns/127 = _
      rw [row_eq]
      rfl
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.LogicalReader
