import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Contract
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealStorage
import VeriTile.Meta.Specification

/-! Kernel correctness for one admitted row, with physical allocation windows,
independent ordinary suffix sums and the complete exterior memory frame. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Correctness
open VeriTile.Triton VeriTile.Torch
noncomputable section

/-- The physical execution uses the same readable source as the FP entry.
Only the mathematical interpretation erases storage/computation dtypes. -/
abbrev source := Kernels.source

theorem kernel_correct (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (c : Config) (chosen : c∈configurations)
    (row : Fin (h.B*h.H)) (before : RealStorage.Heap) :
    Spec.Real (∃ after, RealStorage.execute h p c row.val before=some after ∧
      (∀ off, off<p.backing.input.extent →
        after (Storage.address p.backing.input off)=before (Storage.address p.backing.input off)) ∧
      (∀ t, t<h.T → after (Storage.address p.backing.output (h.outputStrides.address h.H row.val t))=
        MemCell.real (mathematical (RealStorage.input h p before row.val) h.T t)) ∧
      (∀ addr, ¬Storage.writes h p [row] addr → after addr=before addr)) := by
  obtain ⟨after,executed,stable,output,frame⟩ :=
    RealStorage.complete h layout p c (Kernels.configurations_positive c chosen).1 row before
  refine ⟨after,executed,stable,?_,frame⟩
  intro t ht
  rw [output t ht]
  exact congrArg MemCell.real
    (VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula.suffix_filter _ _ _ (Nat.le_of_lt ht))

/-- Full source execution also succeeds from arbitrary register contents and
arbitrary unspecified-value oracles; explicit zero padding makes them irrelevant. -/
theorem source_execution (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (chosen : c∈configurations) (u : BlockState) (row : u.pids 0<h.B*h.H) :
    ∃ v, exec (source h c).eraseDType.toAlgKernel u=some v ∧ RealExecution.Post h u v :=
  RealExecution.source_correct h layout c (Kernels.configurations_positive c chosen).1 u row


#specNonCircular mathematical avoiding [Kernels.source, Kernels.indexCommuted]
#axiomsClean kernel_correct
#axiomsClean source_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Correctness
