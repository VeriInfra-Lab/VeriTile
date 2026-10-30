import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FastLayernorm.FPEquiv
open VeriTile Triton FP.Structural KernelNative
open scoped VeriTile.Spec
noncomputable section

def original (R : Reductions) (stage : IOPair.Stage) :=
  ProgramIO.Implementation.original (IOPair.io (algebra R) stage) (fun _ => True)
def candidate (R : Reductions) (stage : IOPair.Stage) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant (algebra R) stage) (fun _ => True)

/-- Actual complete source pairs at each of the three established IO scopes.
Forward/backward use the retained X row-index candidate. Affine removes the
per-row inactive-column select while retaining both load masks and every
floating operation. Both executions are safe and succeed, active typed outputs
agree and each complete exterior frame holds, for all native inputs. -/
specification kernel_equivalent (R : Reductions) (stage : IOPair.Stage) :
    original R stage ≡[[]] candidate R stage := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.equivalent (algebra R) stage

#axiomsClean kernel_equivalent
end
end VeriTile.Bench.Repairs.FastLayernorm.FPEquiv
