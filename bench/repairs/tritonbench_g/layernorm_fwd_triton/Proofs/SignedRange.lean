import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.SignedRange
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

/-- The exact signed induction variable follows the source. A nonnegative
launch interval has the same ordered recurrence indices without an FP law. -/
theorem values {α β : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg idx .int [] (fun _ => (n : Int))) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t)
    (h : I cur initial s) :
    ∃ final, rangeInt M idx (cur : Int) (stop : Int) stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  by_cases hz : stride = 0
  · exact ⟨s, by simp [rangeInt, hz], by simpa [foldRange, hz] using hdone cur initial s h⟩
  by_cases hc : cur < stop
  · obtain ⟨next, he, hn⟩ := hstep cur initial s h hc
    obtain ⟨final, hf, hd⟩ := values M idx body update I Done
      (cur+stride) stop stride (update cur initial) next hstep hdone hn
    refine ⟨final, ?_, ?_⟩
    · rw [rangeInt]
      simpa only [hz, Int.ofNat_lt, hc, if_false, if_true, he, Option.bind_some,
        Int.natCast_add] using hf
    · rw [foldRange]
      simpa only [hz, hc, if_false, if_true] using hd
  · exact ⟨s, by simp [rangeInt, hz, hc], by simpa [foldRange, hz, hc] using hdone cur initial s h⟩
termination_by stop-cur
decreasing_by omega

#axiomsClean values
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.SignedRange
