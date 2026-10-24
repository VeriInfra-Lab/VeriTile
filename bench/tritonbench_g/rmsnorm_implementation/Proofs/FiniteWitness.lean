import bench.tritonbench_g.rmsnorm_implementation.Proofs.FullContracts

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.FiniteWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

/-- Finite disjoint allocations for two nonzero program IDs, nonunit strides,
two iterations in each loop, a partial tail and raw integer sentinels. -/
def allocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("X", 18), ("W", 10), ("O", 32)]

def initial : BlockState where
  mem := fun r o => if r = "X" then .real 2
    else if r = "W" ∧ o < 9 then .real 3 else .of .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 ∨ ax = 1 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

def kernel (input weight output : Storage) (v : Bool) : ComputeKernel :=
  typedKernel input weight output v "X" "W" "O" 8 4 2 3 16 8 2 3 2 5

theorem disjoint : allocation.Disjoint :=
  FlatAlloc.ofList_disjoint _ _ (by decide)

theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 :=
  FlatAlloc.ofList_closed _ _

theorem input_bounds (t : Fin (rmsNumSteps 3 2)) (j : Fin 2)
    (h : t.val * 2 + j.val < 3) :
    initial.pids 0 * 8 + initial.pids 1 * 4 + (t.val * 2 + j.val) * 2 < allocation.extent "X" := by
  change 12 + (t.val * 2 + j.val) * 2 < 18
  omega

theorem weight_bounds (t : Fin (rmsNumSteps 3 2)) (j : Fin 2)
    (h : t.val * 2 + j.val < 3) :
    (t.val * 2 + j.val) * 3 < allocation.extent "W" := by
  change (t.val * 2 + j.val) * 3 < 10
  omega

theorem output_bounds (t : Fin (rmsNumSteps 3 2)) (j : Fin 2)
    (h : t.val * 2 + j.val < 3) :
    initial.pids 0 * 16 + initial.pids 1 * 8 + (t.val * 2 + j.val) * 2 < allocation.extent "O" := by
  change 24 + (t.val * 2 + j.val) * 2 < 32
  omega

@[simp] theorem pid_zero : initial.pids 0 = 1 := rfl
@[simp] theorem pid_one : initial.pids 1 = 1 := rfl
@[simp] theorem read_x (o : Nat) : initial.readMem "X" o = 2 := rfl

theorem read_w (t : Fin (rmsNumSteps 3 2)) (j : Fin 2)
    (h : t.val * 2 + j.val < 3) :
    initial.readMem "W" ((t.val * 2 + j.val) * 3) = 3 := by
  have hw : (t.val * 2 + j.val) * 3 < 9 := by omega
  simp [initial, BlockState.readMem, hw]

/-- This evaluates only the independent guarded input-memory formula. -/
theorem formula (t : Fin (rmsNumSteps 3 2)) (j : Fin 2) :
    rmsImplStreamSpec 3 2 5 (fun _ _ => 2) (fun _ _ => 3) t j = 2 := by
  have hs : rmsImplStreamSumSq 3 2 (fun _ _ => 2) = 12 := by
    change (∑ u : Fin 2, ∑ e : Fin 2,
      if u.val * 2 + e.val < 3 then (2 : ℝ) ^ 2 else 0) = 12
    norm_num [Fin.sum_univ_succ]
  norm_num [rmsImplStreamSpec, hs]

/-- Every cell outside the streamed write set retains its full original cell,
including non-real values and unallocated flat addresses. -/
def Frame (s : BlockState) : Prop :=
  ∀ r o, (r ≠ allocation.flat ∨
    ∀ (t : Fin (rmsNumSteps 3 2)) (j : Fin 2), t.val * 2 + j.val < 3 →
      o ≠ allocation.addr "O" (24 + (t.val * 2 + j.val) * 2)) →
    s.mem r o = (allocation.flattenState initial).mem r o

/-- All finite allocation and access premises are discharged for all 216
profiles and both variants; safety, successful flat execution and the complete
raw frame refer to the same full two-pass kernel. -/
theorem contract (input weight output : Storage) (v : Bool) :
    (kernel input weight output v).eraseDType.toAlgKernel.TraceSafeR .triv
      allocation.extent initial ∧
    ∃ s, exec (allocation.flattenKernel (kernel input weight output v).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some s ∧
      (∀ (t : Fin (rmsNumSteps 3 2)) (j : Fin 2), t.val * 2 + j.val < 3 →
        s.readMem allocation.flat (allocation.addr "O" (24 + (t.val * 2 + j.val) * 2)) = 2) ∧
      Frame s := by
  have hfull := typed_full_exact_io input weight output v
    "X" "W" "O" 8 4 2 3 16 8 2 3 2 5 (by decide) (by decide) (by decide) (by decide)
  have hs := hfull.1 allocation.extent initial input_bounds weight_bounds output_bounds
  obtain ⟨s, he, hv, hf⟩ := hfull.2 allocation disjoint rfl closed 1 1
    (fun _ _ => 2) (fun _ _ => 3) initial rfl rfl rfl
    input_bounds weight_bounds output_bounds (fun _ _ _ => rfl) read_w
  refine ⟨hs, s, he, ?_, ?_⟩
  · intro t j h
    simpa only [typedIO, rmsnormImplementationKernelIO, Nat.one_mul, formula] using hv t j h
  · simpa only [Frame, typedIO, rmsnormImplementationKernelIO, Nat.one_mul] using hf

theorem sentinels {s : BlockState} (hf : Frame s) :
    s.mem allocation.flat (allocation.addr "O" 25) = MemCell.of .int 37 ∧
    s.mem allocation.flat (allocation.addr "O" 30) = MemCell.of .int 37 ∧
    s.mem allocation.flat (allocation.addr "W" 9) = MemCell.of .int 37 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro t j h
      change 28 + 25 ≠ 28 + (24 + (t.val * 2 + j.val) * 2)
      omega
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro t j h
      change 28 + 30 ≠ 28 + (24 + (t.val * 2 + j.val) * 2)
      omega
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro t j h
      change 18 + 9 ≠ 28 + (24 + (t.val * 2 + j.val) * 2)
      omega

/-- One successful flat run simultaneously gives all three nonzero outputs,
the stride-gap and masked-tail sentinels, and the full raw frame. -/
theorem cells (input weight output : Storage) (v : Bool) :
    ∃ s, exec (allocation.flattenKernel (kernel input weight output v).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some s ∧
      s.readMem allocation.flat (allocation.addr "O" 24) = 2 ∧
      s.readMem allocation.flat (allocation.addr "O" 26) = 2 ∧
      s.readMem allocation.flat (allocation.addr "O" 28) = 2 ∧
      s.mem allocation.flat (allocation.addr "O" 25) = MemCell.of .int 37 ∧
      s.mem allocation.flat (allocation.addr "O" 30) = MemCell.of .int 37 ∧
      s.mem allocation.flat (allocation.addr "W" 9) = MemCell.of .int 37 ∧
      Frame s := by
  obtain ⟨_, s, he, hv, hf⟩ := contract input weight output v
  exact ⟨s, he, hv ⟨0, by decide⟩ 0 (by decide), hv ⟨0, by decide⟩ 1 (by decide),
    hv ⟨1, by decide⟩ 0 (by decide),
    (sentinels hf).1, (sentinels hf).2.1, (sentinels hf).2.2, hf⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.FiniteWitness
