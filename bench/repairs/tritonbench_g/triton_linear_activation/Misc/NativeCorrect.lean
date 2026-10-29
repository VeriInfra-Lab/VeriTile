import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.NativeInterface

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.TritonLinearActivation.FPEquiv
open VeriTile.Triton FP.Structural KernelSource KernelContract KernelNativeModel
noncomputable section
set_option autoImplicit false

/-- Actual selected source, typed IO, raw input encoding and native graph
observation. Compiler policy and the fixed ISA/libdevice primitive model are
explicit model boundaries, not floating transformation assumptions. -/
def linearIO := LinearIO.nativeIO

/-- Independent logical native recurrence, retaining padded typed loads,
source-order casts/additions, and every configured instruction lane. -/
def reference := LinearIO.nativeTarget

/-- Exact complete-source observation for all input words and both source
storage modes. There is no optimization pair or native=Real claim; policies
are not asserted equivalent and no new numerical assumption is introduced. -/
theorem kernel_fp_equiv (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (pid : Nat)
    (positive : 0 < p.blockInner) (strides : CanonicalStrides p) (separate : p.saved ≠ p.output) :
    (linearIO profile policy c mode p pid).Implements (reference profile policy c mode p pid) :=
  LinearIO.native_implements profile policy c mode p pid positive strides separate

#stmtSurfaceSubset kernel_fp_equiv ⊆
  [linearIO, reference, ProgramIO.Implements, LocalProfile, CompilerPolicy, Mode, Parameters,
   Parameters.blockInner, Parameters.saved, Parameters.output, CanonicalStrides, RegionName, Word, Term, instInhabitedTerm]
#specNonCircular reference avoiding [KernelSource.source, FP.Structural.exec]
#axiomsClean kernel_fp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.FPEquiv
