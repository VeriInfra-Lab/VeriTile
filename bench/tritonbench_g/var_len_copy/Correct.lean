import bench.tritonbench_g.var_len_copy.Proofs.LogicalIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.VarLenCopy.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Signed metadata, source segment, destination lanes and all three PID axes.
The logical view fixes the original regional execution judgment. -/
abbrev varLenCopyIO := PublicIO.logicalIO

/-- Complete dynamic copy, including the tail and nonpositive-length branches.
The shared IO relation proves source access safety, successful execution, every
output lane and the exact raw-cell frame outside the destination segment. -/
specification var_len_copy_kernel_real (a : Segment) :
    Spec.Real (varLenCopyIO a ⊨{PublicIO.logicalDomain a} fun xs _ _ _ => xs.source) := by
  exact (PublicIO.logical_iff a).2 (PublicIO.implements a)

#axiomsClean var_len_copy_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.Correct
