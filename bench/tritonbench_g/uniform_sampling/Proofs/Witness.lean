import bench.tritonbench_g.uniform_sampling.Proofs.RealSurface
import bench.tritonbench_g.uniform_sampling.Proofs.TypedExecution

/-! A finite allocation satisfying the complete real I/O contract at the
same nonempty, masked, negative-offset configuration as the typed witness.
Only input/domain facts are computed; no random output is precomputed.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Witness
open VeriTile.Triton TypedSource
noncomputable section

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["out"]
  base := fun _ => 19
  extent := fun r => if r = "out" then 4095 else 0

def initial : BlockState where
  mem := fun _ _ => .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

theorem allocation_disjoint : allocation.Disjoint := by
  simp [FlatAlloc.Disjoint, allocation]

theorem allocation_complete : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r hr
  have hn : r ≠ ("out" : RegionName) := by simpa [allocation] using hr
  simp [allocation, hn]

theorem output_bounds : FlatMemory.Bounds "out" 4095 1024 0 allocation.extent :=
  FlatMemory.bounds_of_extent "out" 4095 1024 0 allocation.extent (by simp [allocation])

theorem real_nonempty (storage : Storage) (swapped : Bool) :
    ∃ v,
      exec (allocation.flattenKernel
        (source storage swapped "out" 4095 (-1) (-1) 0 1 1024 TypedExecution.block_legal).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some v ∧
      (∀ f : Fin 4, ∀ j : Fin 1024, SourceDomain.outputIndex 0 1024 f j < 4095 →
        v.mem allocation.flat (allocation.addr "out" (SourceDomain.outputIndex 0 1024 f j)) =
          MemCell.real (Formula.expected 1024 (-1) (-1) 0 1 (SourceDomain.outputIndex 0 1024 f j))) ∧
      (∀ r o, FlatMemory.Outside allocation 0 "out" 4095 1024 r o →
        v.mem r o = (allocation.flattenState initial).mem r o) := by
  exact RealIO.source_correct storage swapped "out" 4095 (-1) (-1) 0 1 1024
    TypedExecution.block_legal allocation allocation_disjoint allocation_complete initial
    TypedExecution.witness_domain.1 TypedExecution.witness_domain.2.1
    TypedExecution.witness_domain.2.2 TypedExecution.witness_domain.2.2 rfl output_bounds

#axiomsClean allocation_disjoint
#axiomsClean allocation_complete
#axiomsClean output_bounds
#axiomsClean real_nonempty

end
end VeriTile.Bench.TritonBenchG.UniformSampling.Witness
