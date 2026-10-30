import bench.tritonbench_g.matmul_tma.Proofs.SourceSafety
import bench.tritonbench_g.matmul_tma.IO

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem implements {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (injective : Function.Injective (cOffset p))
    (f : (AIndex p → α) → (BIndex p → α) → CIndex p → α)
    (lower : ∀ a b i, M.cast none .fp32 (outputDType half)
      (D.apply (some .fp32) {} (batch := []) a b i) = f a b i) :
    (io M D p half).Implements (formula p half f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded s _
    rw [scheduleSafe_singleton]
    exact SourceSafety.source_safe M D p half ((invocation M D p half).initial s) bounds
      (fun i => bounded.2.1 ⟨0, by simp⟩ i trivial)
      (fun i => bounded.2.1 ⟨1, by simp⟩ i trivial)
      (fun i => bounded.2.2.1 ⟨0, by simp⟩ i trivial)
  · intro vals s binding
    let initial := (invocation M D p half).initial s
    obtain ⟨final, executed, post⟩ := FPValue.correctness M D p half initial injective
    have a : aValues p initial = vals ⟨0, by simp⟩ := by
      funext i
      exact binding ⟨0, by simp⟩ i trivial
    have b : bValues p initial = vals ⟨1, by simp⟩ := by
      funext i
      exact binding ⟨1, by simp⟩ i trivial
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [io, runSchedule_singleton] using executed
    · intro o j _
      fin_cases o
      change mapCell id (final.mem p.c (cOffset p j)) =
        Cell.mk (outputDType half).toTileDType
          (ofFloat (outputDType half) (f (vals 0) (vals 1) j))
      rw [post.1 j]
      simp only [outputValues, a, b, lower]
      cases half <;> rfl
    · intro r offset outside
      apply post.2
      by_cases same : r = p.c
      · right
        intro j equal
        have h := outside.1 ⟨0, by simp⟩ j trivial
        exact h.elim (fun h => h same) (fun h => h equal)
      · exact Or.inl same

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.PublicIO
