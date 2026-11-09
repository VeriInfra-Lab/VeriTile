import VeriTile.Triton.Float.RangeInduction
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.SourceModel

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- A natural mathematical recurrence carried by the actual signed source
counter, without changing source registers, floating operations or failures. -/
theorem range_int_values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg idx .int [] (fun _ => n)) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t)
    (h : I cur initial s) :
    ∃ final, FP.Structural.rangeInt M idx cur stop stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  refine FP.Structural.rangeInt_fold M idx body update I Done cur (stop : Int) stride initial s
    (fun n a => foldRange update n stop stride a) ?_ ?_ hdone h
  · simp only [Int.toNat_natCast]
    intro n a
    simpa only [dite_eq_ite] using foldRange.eq_def update n stop stride a
  · simpa only [Int.toNat_natCast] using hstep

#axiomsClean range_int_values
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Source
