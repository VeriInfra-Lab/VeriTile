import bench.tritonbench_g.var_len_copy.Proofs.KernelFPResults
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.VarLenCopy.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- Integer address commutation uses no floating transformation rule. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original : Program := ⟨Source.originalKernel, []⟩
def commuted : Program := ⟨Source.commutedKernel, []⟩

/-- Both actual source kernels succeed and agree on every typed memory cell.
FP32 values are opaque and unrestricted, including inactive-lane values. -/
specification source_equivalence : original ≡[assumptions] commuted :=
  FPExecution.source_equivalence

#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.FPEquiv
