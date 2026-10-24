import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOSafety
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardIO

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_implements (A : Algebra α) (p : ForwardParameters) (pid : Nat)
    (admissible : KernelExecution.ForwardAdmissible p) :
    (forwardIO A p pid).Implements (forwardGraph A p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact forward_safe A p pid vals bounds hb initial
  · exact forward_result A p pid admissible

theorem backward_implements (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (admissible : KernelReadback.Admissible p) :
    (backwardIO A p pid).Implements (backwardGraph A p pid) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact backward_safe A p pid vals bounds hb initial
  · exact backward_result A p pid admissible

#axiomsClean forward_implements
#axiomsClean backward_implements
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
