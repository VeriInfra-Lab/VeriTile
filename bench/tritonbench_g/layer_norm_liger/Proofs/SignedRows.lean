import VeriTile.Triton.Float.RangeInduction
import bench.tritonbench_g.parallel_attention.Proofs.Sources.Mathematical_FPForwardGraph
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LayerNormLiger.SignedRows
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- Natural mathematical row coordinates refine the original signed range.
The actual body receives an Int counter; no execution premise is assumed. -/
theorem range_values {A B : Type} [Inhabited A] (M : Algebra A)
    (name : RegName) (body : List ComputeStmt) (update : Nat → B → B)
    (I : Nat → B → State A → Prop) (Done : B → State A → Prop)
    (finishRow : Nat)
    (advance : ∀ row a s, I row a s → row < finishRow →
      ∃ t, run M body (s.setReg name .int [] (fun _ => (row : Int))) = some t ∧
        I (row+1) (update row a) t)
    (done : ∀ row a s, I row a s → Done a s)
    (row : Nat) (a : B) (s : State A) (h : I row a s) :
    ∃ t, rangeInt M name (row : Int) (finishRow : Int) 1 body s = some t ∧
      Done (foldRange update row finishRow 1 a) t := by
  refine FP.Structural.rangeInt_fold M name body update I Done row (finishRow : Int) 1 a s
    (fun n a => foldRange update n finishRow 1 a) ?_ ?_ done h
  · simp only [Int.toNat_natCast]
    intro n a
    simpa only [dite_eq_ite] using foldRange.eq_def update n finishRow 1 a
  · simpa only [Int.toNat_natCast] using advance

#axiomsClean range_values
end
end VeriTile.Bench.Repairs.LayerNormLiger.SignedRows
