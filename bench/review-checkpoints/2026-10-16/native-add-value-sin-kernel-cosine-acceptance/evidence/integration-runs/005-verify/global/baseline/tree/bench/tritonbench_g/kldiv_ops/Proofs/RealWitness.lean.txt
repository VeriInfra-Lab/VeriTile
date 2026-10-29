import bench.tritonbench_g.kldiv_ops.Proofs.SourceSafety
import bench.tritonbench_g.kldiv_ops.Proofs.FlatIO

/-! Closed finite-allocation witnesses for the full real contracts. Positive row
length, two loop iterations, nonzero pid and unequal strides are all concrete;
none of allocation validity, input validity or successful execution is assumed. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.RealWitness
open VeriTile.Triton TypedSource

/-- Only active cells are allocated; each final padded input lane lies just outside. -/
def allocation : FlatAlloc := FlatAlloc.ofList "flat" [("Y",8), ("T",10), ("O",12)]

def initial : BlockState where
  mem := fun r _ ↦ MemCell.real (if r = "Y" then 1 else if r = "T" then 2 else 7)
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun _ _ ↦ 0

theorem disjoint : allocation.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)

theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 :=
  FlatAlloc.ofList_closed _ _

theorem y_bounds (k : Fin 3) : initial.pids 0 * 5 + k.val < allocation.extent "Y" := by
  change 5 + k.val < 8
  omega

theorem t_bounds (k : Fin 3) : initial.pids 0 * 7 + k.val < allocation.extent "T" := by
  change 7 + k.val < 10
  omega

theorem o_bounds (k : Fin 3) : initial.pids 0 * 9 + k.val < allocation.extent "O" := by
  change 9 + k.val < 12
  omega

/-- Both target branches and every forward reduction code have a complete
successful run from the explicit finite allocation and concrete input memory. -/
theorem forward_exec (v flag : Bool) (reduction : Nat) :
    ∃ final, exec (allocation.flattenKernel
      (if flag then forwardLog .fp16 .fp32 .bf16 v "Y" "T" "O" 5 7 9 3 2 1 reduction
       else forwardDefault .fp16 .fp32 .bf16 v "Y" "T" "O" 5 7 9 3 2 1 reduction).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final ∧
      ∀ r off, (r ≠ allocation.flat ∨ ∀ k : Fin (if reduction = 0 then 3 else 1),
        off ≠ allocation.addr "O" (9 + k.val)) →
        final.mem r off = (allocation.flattenState initial).mem r off := by
  have he : ∃ final, exec (allocation.flattenKernel
      (FullAlgorithm.forward "Y" "T" "O" 5 7 9 3 2 1 flag reduction))
      (allocation.flattenState initial) = some final ∧
      ∀ r off, (r ≠ allocation.flat ∨ ∀ k : Fin (if reduction = 0 then 3 else 1),
        off ≠ allocation.addr "O" (9 + k.val)) →
        final.mem r off = (allocation.flattenState initial).mem r off := by
    by_cases hr : reduction = 0
    · subst reduction
      obtain ⟨t, he, _, hf⟩ := FlatIO.element_run allocation initial "Y" "T" "O"
        5 7 9 3 2 1 flag disjoint (by simp [allocation]) closed rfl
        (by decide) (by decide) (by decide) y_bounds t_bounds o_bounds
      exact ⟨t, he, hf⟩
    · obtain ⟨t, he, _, hf⟩ := FlatIO.reduction_run allocation initial "Y" "T" "O"
        5 7 9 3 2 1 flag reduction disjoint (by simp [allocation]) closed rfl
        (by decide) hr y_bounds t_bounds (by decide)
      refine ⟨t, he, ?_⟩
      intro r off hno
      apply hf r off
      rcases hno with hne | hmiss
      · exact Or.inl hne
      · exact Or.inr (hmiss ⟨0, by simp [hr]⟩)
  obtain ⟨t, he, hf⟩ := he
  refine ⟨t, ?_, hf⟩
  cases flag
  · exact (Source.forward_default_flat_exec allocation .fp16 .fp32 .bf16 v
      "Y" "T" "O" 5 7 9 3 2 1 reduction (allocation.flattenState initial)).trans he
  · exact (Source.forward_log_flat_exec allocation .fp16 .fp32 .bf16 v
      "Y" "T" "O" 5 7 9 3 2 1 reduction (allocation.flattenState initial)).trans he

theorem backward_exec (v flag : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (if flag then backwardLog .fp32 .bf16 v "T" "O" 7 9 3 2
       else backwardDefault .fp32 .bf16 v "T" "O" 7 9 3 2).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final ∧
      ∀ r off, (r ≠ allocation.flat ∨ ∀ k : Fin 3,
        off ≠ allocation.addr "O" (9 + k.val)) →
        final.mem r off = (allocation.flattenState initial).mem r off := by
  obtain ⟨t, he, _, hf⟩ := FlatIO.backward_run allocation initial "T" "O"
    7 9 3 2 flag disjoint (by simp [allocation]) closed rfl
    (by decide) (by decide) t_bounds o_bounds
  refine ⟨t, ?_, hf⟩
  cases flag
  · exact (Source.backward_default_flat_exec allocation .fp32 .bf16 v
      "T" "O" 7 9 3 2 (allocation.flattenState initial)).trans he
  · exact (Source.backward_log_flat_exec allocation .fp32 .bf16 v
      "T" "O" 7 9 3 2 (allocation.flattenState initial)).trans he

/-- These same witnesses satisfy the actual erased-source trace-safety predicate. -/
theorem forward_safe (v flag : Bool) (reduction : Nat) :
    (if flag then forwardLog .fp16 .fp32 .bf16 v "Y" "T" "O" 5 7 9 3 2 1 reduction
     else forwardDefault .fp16 .fp32 .bf16 v "Y" "T" "O" 5 7 9 3 2 1 reduction).eraseDType.toAlgKernel.TraceSafeR
      .triv allocation.extent initial := by
  cases flag
  · exact Source.forward_default_safe allocation.extent initial .fp16 .fp32 .bf16 v
      "Y" "T" "O" 5 7 9 3 2 1 reduction (by decide) (by intros; decide)
      y_bounds t_bounds (fun _ ↦ o_bounds) (by intros; decide)
  · exact Source.forward_log_safe allocation.extent initial .fp16 .fp32 .bf16 v
      "Y" "T" "O" 5 7 9 3 2 1 reduction (by decide) (by intros; decide)
      y_bounds t_bounds (fun _ ↦ o_bounds) (by intros; decide)

theorem backward_safe (v flag : Bool) :
    (if flag then backwardLog .fp32 .bf16 v "T" "O" 7 9 3 2
     else backwardDefault .fp32 .bf16 v "T" "O" 7 9 3 2).eraseDType.toAlgKernel.TraceSafeR
      .triv allocation.extent initial := by
  cases flag
  · exact Source.backward_default_safe allocation.extent initial .fp32 .bf16 v
      "T" "O" 7 9 3 2 (by decide) t_bounds o_bounds
  · exact Source.backward_log_safe allocation.extent initial .fp32 .bf16 v
      "T" "O" 7 9 3 2 (by decide) t_bounds o_bounds

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.RealWitness
