import bench.tritonbench_g.kldiv_triton.Proofs.Contracts

/-! Finite, disjoint allocations instantiate the complete real contracts.
The nonempty cases use program id one, independent row strides, two loop
iterations and a masked tail. The empty case has zero-sized allocations and
preserves every memory cell. Typed mixed-precision executions are checked
separately in `Proofs/Execution.lean`. -/
namespace VeriTile.Bench.TritonBenchG.KldivTriton.FiniteWitness
open VeriTile.Triton TypedSource
noncomputable section

def initial (flag : Bool) : BlockState where
  mem := fun r _ ↦ if r = "Y" then .real 2
    else if r = "T" then .real (if flag then 0 else 1)
    else .of .int 37
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2

def forwardAllocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("Y", 10), ("T", 12), ("O", 9)]

def backwardAllocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("T", 12), ("O", 9)]

def emptyAllocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("Y", 0), ("T", 0), ("O", 0)]

/-- Both variants and every reduction mode have actual safe flat execution,
the independently specified outputs and the complete untouched-cell frame.
Three columns with block size two make both loop iterations nonempty. -/
theorem forward_execution (v flag : Bool) (reduction : Nat) :
    (forward .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 3 2 flag reduction).eraseDType.toAlgKernel.TraceSafeR
      .triv forwardAllocation.extent (initial flag) ∧
    ∃ t, exec (forwardAllocation.flattenKernel
      (forward .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 3 2 flag reduction).eraseDType.toAlgKernel)
      (forwardAllocation.flattenState (initial flag)) = some t ∧
      (∀ k : Fin (FullMath.outputCount 3 2 reduction),
        t.readMem forwardAllocation.flat (forwardAllocation.addr "O" (5 + k.val)) =
          FullMath.forwardOutput flag
            (fun i ↦ (initial flag).readMem "Y" (7 + i))
            (fun i ↦ (initial flag).readMem "T" (9 + i)) 3 2 reduction k.val) ∧
      (∀ r off, (r ≠ forwardAllocation.flat ∨
          ∀ k : Fin (FullMath.outputCount 3 2 reduction),
            off ≠ forwardAllocation.addr "O" (5 + k.val)) →
        t.mem r off = (forwardAllocation.flattenState (initial flag)).mem r off) := by
  apply Contracts.forward_io .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 3 2 flag reduction
    (by decide) (by decide) (by decide) forwardAllocation (initial flag)
    (FlatAlloc.ofList_disjoint _ _ (by decide)) rfl (FlatAlloc.ofList_closed _ _) rfl
  · intro k
    have hk := k.isLt
    change 7 + k.val < 10
    omega
  · intro k
    have hk := k.isLt
    change 9 + k.val < 12
    omega
  · intro k
    have hk := k.isLt
    change 5 + k.val < 9
    by_cases hr : reduction = 0 <;>
      norm_num [FullMath.outputCount, FullMath.rounds, hr] at hk <;> omega

def backwardKernel (v flag : Bool) : ComputeKernel :=
  if flag then backwardLog .fp64 .fp16 v "T" "O" 9 5 3 2
  else backwardDefault .fp64 .fp16 v "T" "O" 9 5 3 2

/-- Both complete backward branches have an inhabited finite allocation and
preserve the masked neighbor as part of their full memory frame. -/
theorem backward_execution (v flag : Bool) :
    (backwardKernel v flag).eraseDType.toAlgKernel.TraceSafeR
      .triv backwardAllocation.extent (initial flag) ∧
    ∃ t, exec (backwardAllocation.flattenKernel (backwardKernel v flag).eraseDType.toAlgKernel)
      (backwardAllocation.flattenState (initial flag)) = some t ∧
      (∀ k : Fin 3, t.readMem backwardAllocation.flat
        (backwardAllocation.addr "O" (5 + k.val)) =
          FullMath.gradient flag ((initial flag).readMem "T" (9 + k.val))) ∧
      (∀ r off, (r ≠ backwardAllocation.flat ∨
          ∀ k : Fin 3, off ≠ backwardAllocation.addr "O" (5 + k.val)) →
        t.mem r off = (backwardAllocation.flattenState (initial flag)).mem r off) := by
  have ht : ∀ k : Fin 3, (initial flag).pids 0 * 9 + k.val < backwardAllocation.extent "T" := by
    intro k
    have hk := k.isLt
    change 9 + k.val < 12
    omega
  have ho : ∀ k : Fin 3, (initial flag).pids 0 * 5 + k.val < backwardAllocation.extent "O" := by
    intro k
    have hk := k.isLt
    change 5 + k.val < 9
    omega
  cases flag
  · exact Contracts.backward_default_io .fp64 .fp16 v "T" "O" 9 5 3 2
      (by decide) (by decide) backwardAllocation (initial Bool.false)
      (FlatAlloc.ofList_disjoint _ _ (by decide)) rfl (FlatAlloc.ofList_closed _ _) rfl ht ho
  · exact Contracts.backward_log_io .fp64 .fp16 v "T" "O" 9 5 3 2
      (by decide) (by decide) backwardAllocation (initial Bool.true)
      (FlatAlloc.ofList_disjoint _ _ (by decide)) rfl (FlatAlloc.ofList_closed _ _) rfl ht ho

/-- An empty forward row needs no allocated input or output cells, and no
reduction mode emits a scalar or changes any other memory cell. -/
theorem empty_execution (v flag : Bool) (reduction : Nat) :
    ∃ t, exec (emptyAllocation.flattenKernel
      (forward .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 0 2 flag reduction).eraseDType.toAlgKernel)
      (emptyAllocation.flattenState (initial flag)) = some t ∧
      t.mem = (emptyAllocation.flattenState (initial flag)).mem := by
  have h := Contracts.forward_io .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 0 2 flag reduction
    (by decide) (by decide) (by decide) emptyAllocation (initial flag)
    (FlatAlloc.ofList_disjoint _ _ (by decide)) rfl (FlatAlloc.ofList_closed _ _) rfl
    (fun k ↦ Fin.elim0 k) (fun k ↦ Fin.elim0 k)
    (by intro k; have hk := k.isLt; simp at hk)
  obtain ⟨t, he, _, hf⟩ := h.2
  refine ⟨t, he, ?_⟩
  funext r off
  apply hf r off
  right
  intro k
  have hk := k.isLt
  simp at hk

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.FiniteWitness
