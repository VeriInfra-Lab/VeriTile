import bench.tritonbench_g.relu_triton_kernel.Proofs.Structural
import bench.tritonbench_g.relu_triton_kernel.IO

/-! Operational source agreement and frames supporting the public FP IO entry. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.FPObservations
open VeriTile VeriTile.Triton FP.Structural
open FloatingIO
open scoped VeriTile.Spec
noncomputable section

/-- Both actual typed sources succeed and agree on every memory cell; only
private registers differ. Even masked-off operands require no value premise. -/
theorem full_memory_equiv {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real = some cmp)
    (N : Nat) (s : State α) :
    ∃ a b, FP.Structural.exec M (originalKernel N) s = some a ∧
      FP.Structural.exec M (inlinedKernel N) s = some b ∧ a.mem = b.mem := by
  obtain ⟨a, ha, hma⟩ := Structural.source_run M cmp hc "x" "out" 0 0 1 N blockSize s
  obtain ⟨b, hb, hmb⟩ := Structural.fused_run M cmp hc "x" "out" 0 0 1 N blockSize s
  exact ⟨a, b, ha, hb, hma.trans hmb.symm⟩

/-- The output tile and exterior frame follow from the same complete source runs.
The active-mask frame is stronger than the public full-tile FP observation. -/
theorem inlined_observations {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real = some cmp)
    (N : Nat) (s : State α) :
    ∃ a b, exec M (originalKernel N) s = some a ∧
      exec M (inlinedKernel N) s = some b ∧
      (∀ i : Fin blockSize,
        a.mem "out" (s.pids 0 * blockSize + i.val) =
          b.mem "out" (s.pids 0 * blockSize + i.val)) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin blockSize, o ≠ s.pids 0 * blockSize + i.val) →
        a.mem r o = s.mem r o) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : Fin blockSize, o ≠ s.pids 0 * blockSize + i.val) →
        b.mem r o = s.mem r o) := by
  obtain ⟨a, ha, hma⟩ := Structural.source_run M cmp hc "x" "out" 0 0 1 N blockSize s
  obtain ⟨b, hb, hmb⟩ := Structural.fused_run M cmp hc "x" "out" 0 0 1 N blockSize s
  refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
  · intro i
    exact congrFun (congrFun (hma.trans hmb.symm) "out") (s.pids 0 * blockSize + i.val)
  · intro r o outside
    rw [hma]
    apply Structural.outputState_frame
    rcases outside with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ => by simpa using ho i)
  · intro r o outside
    rw [hmb]
    apply Structural.outputState_frame
    rcases outside with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ => by simpa using ho i)

/-- Successful kernel executions, output equality, and each execution's
exterior memory frame. The candidate inlines `result` into the masked store. -/
theorem io_equiv (N : Nat) :
    original N ≡[assumptions] inlined N := by
  apply Spec.FloatingPoint.ofNumerical (lhs := original N) (rhs := inlined N)
    (structural := fun _ _ => False) rfl rfl
  refine ⟨by simp [IO₁PrivateScratch, original, originalIO],
    by simp [IO₁PrivateScratch, inlined, inlinedIO, originalIO], ?_⟩
  intro α _ M D _ hc plans s _
  have hcmp := hc (.le .fp32 .real) (by simp [original])
  simp only [FP.Scheduled.Comparison.Supported] at hcmp
  obtain ⟨cmp, hcmp⟩ := Option.isSome_iff_exists.mp hcmp
  let A := FP.fp32.algebra M plans
  have hcA : A.compareLe (some .fp32) .real = some cmp := hcmp
  simpa only [original, inlined, originalIO, inlinedIO, KernelIO₁.withKernel,
    IO₁Frame, List.not_mem_nil, false_implies, forall_const, implies_true] using
    inlined_observations A cmp hcA N s

/-- Public mathematical and floating proofs refer to the same canonical source. -/
theorem original_source (N : Nat) : (original N).io.kernel = originalKernel N := rfl

#axiomsClean io_equiv
#axiomsClean original_source
#axiomsClean full_memory_equiv
#axiomsClean inlined_observations
end
end VeriTile.Bench.Repairs.ReluTritonKernel.FPObservations
