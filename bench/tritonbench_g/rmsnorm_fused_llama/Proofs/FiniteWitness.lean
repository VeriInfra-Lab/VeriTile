import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.FullContracts

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FiniteWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

/-- Finite disjoint storage includes a nonzero row, its masked tail, and an
unread integer weight sentinel. Both source loops run twice for N=3 and B=2. -/
def allocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("X", 9), ("W", 4), ("Y", 10)]

def initial : BlockState where
  mem := fun r o => if r = "X" then .real 2
    else if r = "W" ∧ o < 3 then .real 3 else .of .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

def kernel (input weight output : Storage) (v : Bool) : ComputeKernel :=
  typedKernel input weight output v "X" "Y" "W" 5 3 2 5

theorem disjoint : allocation.Disjoint :=
  FlatAlloc.ofList_disjoint _ _ (by decide)

theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 :=
  FlatAlloc.ofList_closed _ _

theorem input_bounds (k : Fin 3) :
    initial.pids 0 * 5 + k.val < allocation.extent "X" := by
  change 5 + k.val < 9
  have := k.isLt
  omega

theorem weight_bounds (k : Fin 3) : k.val < allocation.extent "W" := by
  change k.val < 4
  have := k.isLt
  omega

theorem output_bounds (k : Fin 3) :
    initial.pids 0 * 5 + k.val < allocation.extent "Y" := by
  change 5 + k.val < 10
  have := k.isLt
  omega

@[simp] theorem read_x (o : Nat) : initial.readMem "X" o = 2 := rfl

@[simp] theorem read_w (k : Fin 3) : initial.readMem "W" k.val = 3 := by
  simp [initial, BlockState.readMem, k.isLt]

@[simp] theorem row_id : initial.pids 0 = 1 := rfl

/-- The expected value is evaluated from the independent initial-input formula. -/
theorem formula (k : Fin 3) : General.rowSpec initial "X" "W" 5 3 5 k.val = 2 := by
  norm_num [General.rowSpec, Fin.sum_univ_succ]

/-- Every raw cell outside the three active output addresses is retained,
including other dtypes, other regions and unallocated flat addresses. -/
def Frame (t : BlockState) : Prop :=
  ∀ r o, (r ≠ allocation.flat ∨
    ∀ k : Fin 3, o ≠ allocation.addr "Y" (5 + k.val)) →
      t.mem r o = (allocation.flattenState initial).mem r o

/-- All finite allocation, access, zero-undefined and execution premises are
discharged for every floating storage profile and both complete source variants. -/
theorem contract (input weight output : Storage) (v : Bool) :
    (kernel input weight output v).eraseDType.toAlgKernel.TraceSafeR .triv
      allocation.extent initial ∧
    ∃ t, exec (allocation.flattenKernel (kernel input weight output v).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      (∀ k : Fin 3, t.readMem allocation.flat (allocation.addr "Y" (5 + k.val)) = 2) ∧
      Frame t := by
  have hs := General.typed_full_trace_safe input weight output v allocation.extent
    initial "X" "Y" "W" 5 3 2 5 (by decide) (by decide) (by decide)
    input_bounds weight_bounds output_bounds
  obtain ⟨t, he, hv, hf⟩ := General.typed_full_io input weight output v
    "X" "Y" "W" 5 3 2 5 (by decide) (by decide) (by decide)
    allocation initial disjoint rfl closed rfl input_bounds weight_bounds output_bounds
  refine ⟨hs, t, he, ?_, ?_⟩
  · intro k
    simpa only [row_id, Nat.one_mul, formula] using hv k
  · simpa only [Frame, row_id, Nat.one_mul] using hf

theorem sentinels {t : BlockState} (hf : Frame t) :
    t.mem allocation.flat (allocation.addr "Y" 4) = MemCell.of .int 37 ∧
    t.mem allocation.flat (allocation.addr "Y" 8) = MemCell.of .int 37 ∧
    t.mem allocation.flat (allocation.addr "W" 3) = MemCell.of .int 37 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro k
      change 13 + 4 ≠ 13 + (5 + k.val)
      omega
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro k
      change 13 + 8 ≠ 13 + (5 + k.val)
      have := k.isLt
      omega
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro k
      change 9 + 3 ≠ 13 + (5 + k.val)
      omega

/-- The same successful flat execution returns three nonzero values and
retains both adjacent/tail output sentinels and the masked weight sentinel. -/
theorem cells (input weight output : Storage) (v : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel input weight output v).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "Y" 5) = 2 ∧
      t.readMem allocation.flat (allocation.addr "Y" 6) = 2 ∧
      t.readMem allocation.flat (allocation.addr "Y" 7) = 2 ∧
      t.mem allocation.flat (allocation.addr "Y" 4) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "Y" 8) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "W" 3) = MemCell.of .int 37 ∧
      Frame t := by
  obtain ⟨_, t, he, hv, hf⟩ := contract input weight output v
  exact ⟨t, he, hv 0, hv 1, hv 2, (sentinels hf).1,
    (sentinels hf).2.1, (sentinels hf).2.2, hf⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.FiniteWitness
