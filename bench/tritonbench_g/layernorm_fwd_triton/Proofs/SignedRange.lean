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
  by_cases hz : stride = 0
  · exact ⟨s, by simp [FP.Structural.rangeInt, hz], by simpa [foldRange, hz] using hdone cur initial s h⟩
  · by_cases hc : cur < stop
    · obtain ⟨next,he,hn⟩ := hstep cur initial s h hc
      obtain ⟨final,hf,hd⟩ := range_int_values M idx body update I Done
        (cur+stride) stop stride (update cur initial) next hstep hdone hn
      refine ⟨final,?_,?_⟩
      · have hi : (cur : Int) < stop := by exact_mod_cast hc
        rw [FP.Structural.rangeInt]
        simpa only [hz, hi, if_false, if_true, he,
          Option.bind_some, Nat.cast_add] using hf
      · rw [foldRange]
        simpa only [hz,hc,if_false,if_true] using hd
    · exact ⟨s, by simp [FP.Structural.rangeInt,hz,hc],
        by simpa [foldRange,hz,hc] using hdone cur initial s h⟩
termination_by stop-cur
 decreasing_by omega

#axiomsClean range_int_values
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Source
