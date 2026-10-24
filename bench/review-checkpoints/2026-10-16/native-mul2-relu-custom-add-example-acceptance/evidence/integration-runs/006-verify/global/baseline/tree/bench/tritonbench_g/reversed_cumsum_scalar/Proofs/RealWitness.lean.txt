import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealIO

/-! A finite, disjoint allocation with two complete source iterations and
a partial final chunk. This exercises the allocator theorem from empty
registers; only the bridge's required zero unspecified oracle is specialized. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealWitness
open VeriTile.Triton RealValues
noncomputable section

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["s", "o"]
  base := fun r => if r = "s" then 19 else 100
  extent := fun r => if r = "s" ∨ r = "o" then 34 else 0

def initial : BlockState where
  mem := fun r _ => if r = "s" then .real 2 else MemCell.of .bool Bool.true
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

theorem allocation_disjoint : allocation.Disjoint := by
  simp [FlatAlloc.Disjoint, allocation]

theorem allocation_complete : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r hr
  have hn : r ≠ ("s" : RegionName) ∧ r ≠ ("o" : RegionName) := by simpa [allocation] using hr
  simp [allocation, hn.1, hn.2]

theorem bounds : RealSafety.Bounds 17 (initial.pids 0) "s" "o" allocation.extent := by
  constructor <;> intro t ht <;> change 17 + t < 34 <;> omega

theorem real_nonempty (swapped : Bool) (dtype out_dtype : FloatDType) :
    ∃ final, exec (allocation.flattenKernel
      (TypedSource.source swapped dtype out_dtype "s" "o" 17 16).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final ∧
      RealIO.Post allocation 17 "s" "o" initial final := by
  exact RealIO.source_correct swapped allocation allocation_disjoint allocation_complete
    17 16 "s" "o" dtype out_dtype initial (by decide) (by decide) bounds rfl

#axiomsClean allocation_disjoint
#axiomsClean allocation_complete
#axiomsClean bounds
#axiomsClean real_nonempty
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealWitness
