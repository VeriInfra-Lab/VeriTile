import bench.tritonbench_g.rms_rbe_matmul.Proofs.Contracts

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FiniteWitness
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Nonzero batch id, independent strides and a partial K/N tile. The second
K lane is ordinary allocated data even though the logical dimension is one. -/
def allocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("X", 16), ("W", 8), ("R", 5), ("Y", 24)]

def initial : BlockState where
  mem := fun r _ => if r = "X" ∨ r = "W" ∨ r = "R" then MemCell.real 1 else MemCell.of .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

def kernel (input : TypedSource.Storage) (variant : Bool) : ComputeKernel :=
  TypedSource.kernel input Bool.false variant "X" "W" "R" "Y"
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2

@[simp] theorem read_x (o : Nat) : initial.readMem "X" o = 1 := rfl
@[simp] theorem read_w (o : Nat) : initial.readMem "W" o = 1 := rfl
@[simp] theorem read_r (o : Nat) : initial.readMem "R" o = 1 := rfl

theorem disjoint : allocation.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)

theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 :=
  FlatAlloc.ofList_closed _ _

theorem reads : AccessSafety.Reads allocation.extent initial "X" "W" "R"
    1 1 1 1 2 2 11 7 2 3 1 2 := by
  refine ⟨?_, ?_, ?_⟩
  · intro t ht i e
    have ht' : t = 0 := by norm_num [cdiv] at ht; omega
    subst t
    simp only [rowIndex, Nat.mod_one, Nat.zero_mul, Nat.add_zero]
    change 11 + (0 * 2 + e.val) * 2 < 16
    have := e.isLt
    omega
  · intro t ht e j
    have ht' : t = 0 := by norm_num [cdiv] at ht; omega
    subst t
    simp only [colIndex, Nat.mod_one, Nat.zero_mul, Nat.add_zero]
    change (0 * 2 + e.val) * 3 + 0 * 1 < 8
    have := e.isLt
    omega
  · intro t ht e
    have ht' : t = 0 := by norm_num [cdiv] at ht; omega
    subst t
    change (0 * 2 + e.val) * 2 < 5
    have := e.isLt
    omega

theorem writes (i : TileIndex [1, 2]) (_hi : active initial 1 1 1 2 i) :
    outOffset initial 1 1 2 17 5 2 i < allocation.extent "Y" := by
  have hm := i.1.isLt
  have hn := i.2.1.isLt
  change 1 * 17 + (0 * 1 + i.1.val) * 5 + (0 * 2 + i.2.1.val) * 2 < 24
  omega

/-- Both complete variants and all six formal input labels satisfy the finite
flat contract with every premise discharged and a nonempty active output. -/
theorem contract (input : TypedSource.Storage) (variant : Bool) :
    Contracts.ExactIO (kernel input variant) allocation initial "X" "W" "R" "Y"
      1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 2 := by
  refine ⟨?_, FlatIO.half input variant allocation "X" "W" "R" "Y" initial
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2 disjoint closed rfl
    (Geometry.active_injective initial 1 1 1 2 17 5 2 (by decide) (by decide)) reads writes⟩
  exact AccessSafety.complete_half input variant allocation.extent "X" "W" "R" "Y" initial
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2 (by intros; rfl) reads writes

/-- Actual flattened execution returns the nonzero padded-K result one and
retains integer sentinels both before the batch output and in its masked tail. -/
theorem cells (input : TypedSource.Storage) (variant : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel input variant).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "Y" 17) = 1 ∧
      t.mem allocation.flat (allocation.addr "Y" 19) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "Y" 0) = MemCell.of .int 37 := by
  obtain ⟨_, t, he, hv, hf⟩ := contract input variant
  refine ⟨t, he, ?_, ?_, ?_⟩
  · change t.readMem allocation.flat
      (allocation.addr "Y" (outOffset initial 1 1 2 17 5 2 (0, 0, PUnit.unit))) = 1
    rw [hv _ (by decide)]
    norm_num [rmsSpec, normSpec, accPartial, xElem, rmsElem, wElem,
      cdiv, Fin.sum_univ_succ]
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro i hi
      have hm := i.1.isLt
      have hc : i.2.1.val = 0 := by
        have hc := hi.2
        change 0 * 2 + i.2.1.val < 1 at hc
        omega
      have hr : i.1.val = 0 := by omega
      simp [outOffset, rowGlobal, colGlobal, initial, pidM, pidN, cdiv, hr, hc, allocation,
        FlatAlloc.addr, FlatAlloc.ofList]
  · rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro i hi
      have hm := i.1.isLt
      have hc : i.2.1.val = 0 := by
        have hc := hi.2
        change 0 * 2 + i.2.1.val < 1 at hc
        omega
      have hr : i.1.val = 0 := by omega
      simp [outOffset, rowGlobal, colGlobal, initial, pidM, pidN, cdiv, hr, hc, allocation,
        FlatAlloc.addr, FlatAlloc.ofList]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FiniteWitness
