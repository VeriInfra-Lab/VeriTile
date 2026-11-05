import VeriTile.Triton.Float.Structural
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
  by_cases active : row < finishRow
  · obtain ⟨next,executed,inv⟩ := advance row a s h active
    obtain ⟨final,hfinal,result⟩ := range_values M name body update I Done finishRow
      advance done (row+1) (update row a) next inv
    refine ⟨final,?_,?_⟩
    · rw [rangeInt]
      simp only [show (1 : Nat) ≠ 0 by decide, if_false,
        show (row : Int) < finishRow by exact_mod_cast active, if_true,
        executed, Option.bind_some]
      simpa only [Int.natCast_add,Int.natCast_one] using hfinal
    · rw [foldRange]
      simpa only [show (1 : Nat) ≠ 0 by decide,if_false,active,if_true] using result
  · refine ⟨s,?_,?_⟩
    · rw [rangeInt]
      simp only [show (1 : Nat) ≠ 0 by decide,if_false,
        show ¬ (row : Int) < finishRow by exact_mod_cast active]
    · rw [foldRange]
      simpa only [show (1 : Nat) ≠ 0 by decide,if_false,active] using done row a s h
termination_by finishRow-row
decreasing_by omega

#axiomsClean range_values
end
end VeriTile.Bench.Repairs.LayerNormLiger.SignedRows
