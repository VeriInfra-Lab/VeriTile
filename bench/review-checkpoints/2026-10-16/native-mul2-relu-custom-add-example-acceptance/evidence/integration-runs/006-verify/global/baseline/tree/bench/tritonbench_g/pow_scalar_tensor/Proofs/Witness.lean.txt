import bench.tritonbench_g.pow_scalar_tensor.Proofs.Contracts

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.FiniteWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

def allocation : FlatAlloc := FlatAlloc.ofList "flat" [("X", 14), ("Y", 24)]

def exponent (task : Nat) : ℝ := if task = 2 then 2 else if task = 3 then 3 else 1

def initial : BlockState where
  mem := fun r o => if r = "X" then .real (exponent (o / 2)) else .of .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun ax => if ax = 0 then 2 else 1
  undef := fun _ _ => 0

theorem disjoint : allocation.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)

theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 :=
  FlatAlloc.ofList_closed _ _

/-- A complete partial one-tile run with nonzero program id and independent
nonunit strides. No execution, bounds or expected-output facts are assumed. -/
theorem one_tile (output : TypedSource.FloatingOutput) (candidate : Bool) :
    Contracts.ExactIO
      (TypedSource.kernel output Bool.true candidate 2 "X" "Y" 2 3 3 918 2 2)
      allocation initial "Y" 2 (fun i : Fin 2 => (2 + i.val) * 3)
      (fun i => 2 + i.val < 3) (fun i => exponent (2 + i.val)) := by
  apply Contracts.one_tile output candidate allocation 2 "X" "Y" 2 3 3 918 2 2 initial
    (fun i => exponent (2+i.val)) disjoint closed (by decide) (by decide) rfl
  · intro i hi
    simp [initial, taskIndex, BlockState.readMem]
  · intro i hi
    change (2 + i.val) * 2 < 14
    omega
  · intro i hi
    change (2 + i.val) * 3 < 24
    omega

/-- Two actual grid-stride iterations visit tasks 2,3 and 6,7, with task 7
masked. Both source/candidate variants and all formal floating destinations
satisfy finite trace safety, the independent formula and the full cell frame. -/
theorem grid (output : TypedSource.FloatingOutput) (candidate : Bool) :
    Contracts.ExactIO
      (TypedSource.kernel output Bool.false candidate 2 "X" "Y" 2 3 7 918 2 2)
      allocation initial "Y" 2
      (fun p : Fin 2 × Fin 2 => ((1 + p.1.val * 2) * 2 + p.2.val) * 3)
      (fun p => (1 + p.1.val * 2) * 2 + p.2.val < 7)
      (fun p => exponent ((1 + p.1.val * 2) * 2 + p.2.val)) := by
  apply Contracts.grid output candidate allocation 2 "X" "Y" 2 3 7 918 2 2 initial
    (fun j i => exponent ((1+j.val*2)*2+i.val)) disjoint closed (by decide)
    (by decide) rfl (by decide) (by decide)
  · intro j i hi
    simp [initial, taskIndex, BlockState.readMem]
  · intro j i hi
    change ((1 + j.val*2)*2+i.val)*2 < 14
    change (1+j.val*2)*2+i.val < 7 at hi
    omega
  · intro j i hi
    change ((1 + j.val*2)*2+i.val)*3 < 24
    change (1+j.val*2)*2+i.val < 7 at hi
    omega

/-- Concrete readbacks of the full grid source: four, eight and two, with the
masked tail and skipped program's output retaining their integer sentinels. -/
theorem grid_values (output : TypedSource.FloatingOutput) (candidate : Bool) :
    ∃ t, exec (allocation.flattenKernel
      (TypedSource.kernel output Bool.false candidate 2 "X" "Y" 2 3 7 918 2 2).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "Y" 6) = 4 ∧
      t.readMem allocation.flat (allocation.addr "Y" 9) = 8 ∧
      t.readMem allocation.flat (allocation.addr "Y" 18) = 2 ∧
      t.mem allocation.flat (allocation.addr "Y" 21) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "Y" 12) = MemCell.of .int 37 := by
  obtain ⟨_, t, he, hv, hf⟩ := grid output candidate
  refine ⟨t, he, ?_, ?_, ?_, ?_, ?_⟩
  · have h := hv (0,0) (by decide)
    norm_num [exponent, Real.rpow_two] at h
    exact h
  · have h := hv (0,1) (by decide)
    norm_num [exponent, show (3 : ℝ) = (3 : ℕ) from rfl, Real.rpow_natCast] at h
    exact h
  · simpa [exponent] using hv (1,0) (by decide)
  all_goals
    rw [hf allocation.flat _ (Or.inr ?_),
      allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
    · rfl
    · intro p hp heq
      have ha : Function.Injective (allocation.addr "Y") := by
        intro a b hab
        exact Nat.add_left_cancel hab
      have h := ha heq
      rcases p with ⟨j,i⟩
      dsimp at hp h
      omega

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.FiniteWitness
