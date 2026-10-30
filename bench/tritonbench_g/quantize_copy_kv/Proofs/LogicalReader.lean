import bench.tritonbench_g.quantize_copy_kv.Proofs.ReaderBefore
import bench.tritonbench_g.quantize_copy_kv.IO
namespace VeriTile.Bench.Repairs.QuantizeCopyKv.LogicalReader
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
def formula (p : Parameters) (count : Nat) (xs : LogicalView.Inputs p count) : LogicalView.Outputs p count where
  code := fun t h c =>
    let x := LogicalView.row xs t h
    if mathematicalMaximum x p.width=0 then 0 else
      let z := xs.values t h c/(mathematicalMaximum x p.width/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt
  scale := fun t h => mathematicalMaximum (LogicalView.row xs t h) p.width/127

/-- The complete prior judgment is equivalent after bijective tensor reindexing. -/
theorem judgment_iff (rest : Algebra ℝ) (p : Parameters) (count : Nat) (dest : Nat → Nat) (version : Bool) :
    (LogicalView.quantizeIO rest p count dest version).ImplementsOn (LogicalView.Metadata dest) (formula p count) ↔
    (ReaderBefore.quantizeIO rest p count dest version).ImplementsOn (Interface.metadataMatches p count dest) (ReaderBefore.mathematical p count) := by
  unfold LogicalProgramIO.ImplementsOn LogicalProgramIO.lift
  apply Iff.of_eq
  congr 1
  funext vals o j
  change Interface.Inputs ℝ p count at vals
  have row_eq (t : Fin count) (h : Fin p.heads) :
      LogicalView.row (LogicalView.inputEquiv p count vals) t h = Interface.payload vals t.val h.val := by
    funext c
    change (if hc : c<p.width then vals (1 : Fin 2) (t,h,⟨c,hc⟩,()) else (0 : ℝ)) = _
    simp only [Interface.payload,dif_pos t.isLt,dif_pos h.isLt]
    split <;> rfl
  fin_cases o
  · change (formula p count (LogicalView.inputEquiv p count vals)).code j.1 j.2.1 j.2.2.1 = _
    dsimp only [formula]
    rw [row_eq]
    have element_eq : (LogicalView.inputEquiv p count vals).values j.1 j.2.1 j.2.2.1 =
        Interface.payload vals j.1.val j.2.1.val j.2.2.1.val := by
      change vals (1 : Fin 2) (j.1,j.2.1,j.2.2.1,()) = _
      simp only [Interface.payload,dif_pos j.1.isLt,dif_pos j.2.1.isLt,dif_pos j.2.2.1.isLt]
      rfl
    rw [element_eq]
    rfl
  · change mathematicalMaximum (LogicalView.row (LogicalView.inputEquiv p count vals) j.1 j.2.1) p.width/127 = _
    rw [row_eq]
    rfl
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.LogicalReader
