import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.IO
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.IOResults
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.LogicalIO
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem forward_iff (d : Host.Dimensions) :
    (forward d ⊨ fun a h t c => Target.output (a.query h) (a.key h) (a.value h) t c) ↔
    (forwardIO d).Implements (forwardFormula d) := by
  change (forwardIO d).Implements _ ↔ _
  apply Iff.of_eq
  congr 1

theorem backward_iff (d : Host.Dimensions) :
    (io d ⊨ fun a => ⟨
      fun h t c => Target.dq (a.key h) (a.value h) (a.incoming h) t c,
      fun h t c => Target.dk (a.query h) (a.value h) (a.incoming h) t c,
      fun h t c => Target.dv (a.query h) (a.key h) (a.incoming h) t c⟩) ↔
    (backwardIO d).Implements (backwardFormula d) := by
  change (backwardIO d).Implements _ ↔ _
  apply Iff.of_eq
  congr 1

theorem forward_implements (d : Host.Dimensions) (geometry : Host.Contract d) :
    forward d ⊨ fun a h t c => Target.output (a.query h) (a.key h) (a.value h) t c := by
  apply (forward_iff d).mpr
  apply ProgramIO.Implements.intro
  · intro x bounds hb initial binding
    exact IOProof.forward_safety d geometry x bounds hb initial
  · exact IOProof.forward_result d geometry

theorem backward_implements (d : Host.Dimensions) (geometry : Host.Contract d) :
    io d ⊨ fun a => ⟨
      fun h t c => Target.dq (a.key h) (a.value h) (a.incoming h) t c,
      fun h t c => Target.dk (a.query h) (a.value h) (a.incoming h) t c,
      fun h t c => Target.dv (a.query h) (a.key h) (a.incoming h) t c⟩ := by
  apply (backward_iff d).mpr
  apply ProgramIO.Implements.intro
  · intro x bounds hb initial binding
    exact IOProof.backward_safety d x bounds hb initial
  · exact IOProof.backward_result d geometry
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.LogicalIO
