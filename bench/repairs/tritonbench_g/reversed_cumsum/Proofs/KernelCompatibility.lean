import bench.repairs.tritonbench_g.reversed_cumsum.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.KernelContract
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealStorage
import VeriTile.Meta.Specification

/-! Preserved physical-memory consequence of the same source execution used
by public IO. The byte-addressed element-cell frame and input allocation
stability remain available to optional Tensor/placement consumers. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.KernelCompatibility
open VeriTile.Triton VeriTile.Torch
noncomputable section

abbrev source := Kernels.source

theorem physical_kernel_correct (h : Call) (distinct : h.input≠h.output)
    (p : Storage.Placement h) (c : Config) (chosen : c∈configurations)
    (program : Geometry.Program h) (before : RealStorage.Heap) :
    (∃ after, RealStorage.execute h p c program before=some after ∧
      (∀ off, off<p.backing.input.extent →
        after (Storage.address p.backing.input off)=before (Storage.address p.backing.input off)) ∧
      (∀ t, ∀ j : Fin BS, t<h.T → program.2.val*BS+j.val<h.S →
        after (Storage.address p.backing.output (Geometry.address h program.1.val t (program.2.val*BS+j.val)))=
          MemCell.real (mathematical (fun k => RealStorage.input h p before program.1.val k (program.2.val*BS+j.val)) h.T t)) ∧
      (∀ addr, ¬Storage.writes h p [program] addr → after addr=before addr)) := by
  obtain ⟨after,executed,stable,output,frame⟩ :=
    RealStorage.complete h distinct p c (Kernels.configurations_positive c chosen).1 program before
  refine ⟨after,executed,stable,?_,frame⟩
  intro t j ht hs
  rw [output t j ht hs]
  exact congrArg MemCell.real
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffix_filter _ _ _ (Nat.le_of_lt ht))

#specNonCircular mathematical avoiding [Kernels.source, Kernels.compareRight]
#axiomsClean physical_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.KernelCompatibility
