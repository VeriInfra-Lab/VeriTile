import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.quantize_kv_copy.IO
namespace VeriTile.Bench.Repairs.QuantizeKvCopy.LogicalReader
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
def formula (p : Parameters) (n : Nat) (xs : LogicalView.Inputs p n) : LogicalView.Outputs p n where
  code := fun t h c =>
    let x := LogicalView.row xs t h
    if mathematicalMaximum x 8=0 then 0 else
      let z := xs.values t h c/(mathematicalMaximum x 8/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt
  scale := fun t h => mathematicalMaximum (LogicalView.row xs t h) 8/127

/-- The complete prior judgment is equivalent after bijective tensor reindexing. -/
theorem judgment_iff (p : Parameters) (n : Nat) (dest : Nat → Nat) :
    (LogicalView.quantizeIO p n dest).ImplementsOn (LogicalView.Metadata dest) (formula p n) ↔
    (ReaderBefore.quantizeIO p n dest).ImplementsOn (PublicIO.Metadata dest)
      (fun values => PublicIO.result
        (fun pid g c => let x := PublicIO.elements values pid g
          ((if mathematicalMaximum x 8=0 then (0 : BitVec 8) else
            let z := x c/(mathematicalMaximum x 8/127)
            BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z))).toInt)
        (fun pid g => mathematicalMaximum (PublicIO.elements values pid g) 8/127)) := by
  unfold LogicalProgramIO.ImplementsOn LogicalProgramIO.lift
  apply Iff.of_eq
  congr 1
  funext vals o j
  change PublicIO.Inputs ℝ p n at vals
  have row_eq (t : Fin n) (h : Fin p.groups) :
      LogicalView.row (LogicalView.inputEquiv p n vals) t h = PublicIO.elements vals t.val h.val := by
    funext c
    change (if hc : c<8 then vals (0 : Fin 2) (t,h,⟨c,hc⟩,()) else (0 : ℝ)) = _
    simp only [PublicIO.elements,dif_pos t.isLt,dif_pos h.isLt]
    split <;> rfl
  fin_cases o
  · change (formula p n (LogicalView.inputEquiv p n vals)).code j.1 j.2.1 j.2.2.1 = _
    dsimp only [formula,PublicIO.result]
    rw [row_eq]
    have element_eq : (LogicalView.inputEquiv p n vals).values j.1 j.2.1 j.2.2.1 =
        PublicIO.elements vals j.1.val j.2.1.val j.2.2.1.val := by
      change vals (0 : Fin 2) (j.1,j.2.1,j.2.2.1,()) = _
      simp only [PublicIO.elements,dif_pos j.1.isLt,dif_pos j.2.1.isLt,dif_pos j.2.2.1.isLt]
      rfl
    rw [element_eq]
    split <;> rfl
  · change mathematicalMaximum (LogicalView.row (LogicalView.inputEquiv p n vals) j.1 j.2.1) 8/127 = _
    rw [row_eq]
    rfl
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.LogicalReader
