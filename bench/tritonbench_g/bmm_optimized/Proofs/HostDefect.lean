import bench.tritonbench_g.bmm_optimized.Proofs.RealContract
import VeriTile.Triton.Memory.FlattenFrame

/-! A host logical-view defect independent of omitted `other`. A contiguous
lazy-negative FP32 A has physical +1 but logical -1; `.contiguous()` returns
that same view. With M=N=128,K=32 every source configuration has all three
divisibility flags true. The selected 128×128×32/group2 configuration has
exactly one program and no masked loads. This module proves its complete
finite-memory real execution stores +32, whereas the logical product is -32.
The exact original-host/real-Torch pointer witness is in the task checkpoint.
No native GPU or IEEE-emulation claim is made. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.HostDefect
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000

/-- Tile/group configuration at source list indices 5 and 11. -/
def parameters : RealProgram.Parameters :=
  ⟨"A", "B", "O", 128, 128, 32, 2, Bool.true, Bool.true, Bool.true⟩

def initial : BlockState where
  mem := fun r _ => MemCell.real (if r = "O" then 9 else 1)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

def allocation : FlatAlloc :=
  FlatAlloc.ofList "arena" [("A",4096), ("B",4096), ("O",16384)]

def logicalA (_m : Fin 128) (_k : Fin 32) : ℝ := -1
def logicalB (_k : Fin 32) (_n : Fin 128) : ℝ := 1
def logicalExpected (m n : Fin 128) : ℝ :=
  ∑ k : Fin 32, logicalA m k * logicalB k n

theorem logical_expected (m n : Fin 128) : logicalExpected m n = -32 := by
  norm_num [logicalExpected, logicalA, logicalB]

theorem host : RealMemory.Host 128 128 32 parameters := by
  constructor <;> norm_num [parameters]

theorem launch : RealMemory.Launch initial 128 128 parameters := by
  constructor <;> norm_num [initial, parameters, BmmChunkBwd.RealMath.blocks]

theorem view_launch (swapped : Bool) :
    (RealViews.io swapped 128 128 32 parameters).Launch initial := by
  constructor <;> norm_num [RealViews.io, initial, parameters, BmmChunkBwd.RealMath.blocks]

theorem row_zero (i : Fin 128) : RealSetup.rows initial 128 parameters i = i.val := by
  simp [RealSetup.rows, RealSetup.tileM, RealGrid.rowTile, RealGrid.groupSize, initial, parameters]

theorem column_zero (j : Fin 128) : RealSetup.columns initial 128 parameters j = j.val := by
  simp [RealSetup.columns, RealSetup.tileN, RealGrid.columnTile, RealGrid.groupSize, initial, parameters]

theorem formula (i : TileIndex [128,128]) :
    RealMemory.formula initial 128 128 parameters i = 32 := by
  simp [RealMemory.formula, RealMath.contraction, RealLoop.aEntry, RealLoop.bEntry,
    RealSetup.aBase, RealSetup.bBase, initial, parameters, BlockState.readMem]

theorem allocation_disjoint : allocation.Disjoint :=
  FlatAlloc.ofList_disjoint _ _ (by decide)

theorem bounds (swapped : Bool) :
    RealIO.Bounds (RealViews.io swapped 128 128 32 parameters) initial allocation.extent := by
  constructor
  · rintro ⟨i,k,⟨⟩⟩ _
    have hk : k.val < 32 := k.isLt
    norm_num [RealViews.io, RealViews.row, RealGrid.rowTile, RealGrid.groupSize,
      BmmChunkBwd.RealMath.blocks, initial, parameters, allocation, FlatAlloc.ofList,
      FlatAlloc.listExtent]
    omega
  · rintro ⟨k,j,⟨⟩⟩ _
    have hk : k.val < 32 := k.isLt
    norm_num [RealViews.io, RealViews.column, RealGrid.columnTile, RealGrid.groupSize,
      BmmChunkBwd.RealMath.blocks, initial, parameters, allocation, FlatAlloc.ofList,
      FlatAlloc.listExtent]
    omega
  · rintro ⟨i,j,⟨⟩⟩ _
    norm_num [RealViews.io, RealViews.row, RealViews.column, RealGrid.rowTile,
      RealGrid.columnTile, RealGrid.groupSize, BmmChunkBwd.RealMath.blocks, initial,
      parameters, allocation, FlatAlloc.ofList, FlatAlloc.listExtent]
    simp only [String.reduceEq, ↓reduceIte]
    omega

/-- Full source execution, finite allocations and actual stored first cell.
All flags are true, so no unspecified masked-load value is involved. -/
theorem finite_source_executes (swapped : Bool) :
    ∃ t, exec (allocation.flattenKernel (RealProgram.kernel swapped 128 128 32 parameters).toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "O" 0) = 32 ∧
      (∀ r a, r ≠ allocation.flat ∨
        (∀ o, o < 16384 → a ≠ allocation.addr "O" o) →
        t.mem r a = (allocation.flattenState initial).mem r a) := by
  obtain ⟨t, he, hv, hf⟩ := RealMemory.executes swapped initial 128 128 32 parameters
    host launch (by intros; rfl)
  have hcell : t.mem "O" 0 = MemCell.real 32 := by
    have h := hv (0,0,PUnit.unit)
      (by simp only [RealMemory.active, row_zero, column_zero]; decide)
    rw [formula] at h
    change t.mem "O" (RealSetup.rows initial 128 parameters 0 * 128 +
      RealSetup.columns initial 128 parameters 0) = MemCell.real 32 at h
    simpa only [row_zero, column_zero, Fin.val_zero, Nat.zero_mul, Nat.zero_add] using h
  refine ⟨allocation.flattenState t, ?_, ?_, ?_⟩
  · have hs : (RealProgram.kernel swapped 128 128 32 parameters).toAlgKernel.TraceSafe
        allocation.extent initial :=
      RealViews.safe swapped initial 128 128 32 parameters host (view_launch swapped)
        rfl allocation.extent (bounds swapped)
    rw [allocation.exec_flatten allocation_disjoint
      (FlatAlloc.ofList_closed _ _) (RealProgram.kernel swapped 128 128 32 parameters).toAlgKernel
      initial hs (RealPureSafety.flatten_ok swapped 128 128 32 parameters) rfl, he]
    rfl
  · rw [allocation.flattenState_readMem allocation_disjoint t
      (by simp [allocation, FlatAlloc.ofList])
      (by decide)]
    simp [BlockState.readMem, hcell]
  · intro r a hnot
    refine allocation.flattenState_frame initial t
      (fun region off => region = "O" ∧ off < 16384) ?_ r a ?_
    · intro region off hn
      apply hf region off
      intro i hi heq
      apply hn
      refine ⟨heq.1.symm, ?_⟩
      have hm := i.1.isLt
      have hn := i.2.1.isLt
      simp only [RealMemory.address, row_zero, column_zero] at heq
      simp only [RealSetup.oBase, initial, parameters, Nat.zero_mul, Nat.zero_add] at heq
      omega
    · rcases hnot with hreg | hoff
      · exact Or.inl hreg
      · right
        rintro region off ⟨rfl, ho⟩
        exact hoff off ho

/-- The expected logical input product differs from the proved raw execution. -/
theorem source_disagrees_with_logical_input :
    ∃ t, exec (allocation.flattenKernel (RealProgram.kernel Bool.false 128 128 32 parameters).toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "O" 0) ≠ logicalExpected 0 0 := by
  obtain ⟨t, he, hv, _⟩ := finite_source_executes Bool.false
  refine ⟨t, he, ?_⟩
  rw [hv, logical_expected]
  norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.HostDefect
