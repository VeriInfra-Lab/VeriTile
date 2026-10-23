import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.IOComplete
import VeriTile.Meta.Specification

/-! Exact source-to-native FP32 observation for every explicit FP.Config.
This task has no optimization rewrite: FP.mul is the concrete primitive model,
including all words, overflow, subnormals, infinities and NaNs. No numerical
identity is assumed and no elementary bit-pattern fact is deferred to experiment.
The selected RNE/no-FTZ profile is an optional ordinary bridge consequence. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPEquiv
open VeriTile Triton FP.Structural Interface
noncomputable section

def nativeIO (profile : FP.Config) (src dst : RegionName) (n B : Nat)
    (order : List Nat) : ProgramIO KernelArithmetic.FPWord KernelArithmetic.FPWord :=
  io (KernelArithmetic.nativeAlgebra profile) src dst n B order

/-- Independent ordered FP32 multiply by the exact word 0x77800000. -/
def nativeFormula (profile : FP.Config) (src dst : RegionName) (n : Nat) :=
  formula (KernelArithmetic.native profile) src dst n

/-- Exact native output for the whole source grid, any complete program order,
all FP32 input words and both in-place and separate output regions. -/
theorem kernel_fp_profile (profile : FP.Config) (src dst : RegionName)
    (n B : Nat) (positive : 0<B) (order : List Nat)
    (schedule : order.Perm (List.range (grid n B))) :
    (nativeIO profile src dst n B order).Implements (nativeFormula profile src dst n) :=
  IOComplete.native_implements profile src dst n B positive order schedule

#specNonCircular nativeFormula avoiding [candidateKernel,runGrid,FP.Structural.exec,runSchedule]
#stmtSurfaceSubset kernel_fp_profile ⊆
  [FP.Config,RegionName,grid,nativeIO,ProgramIO.Implements,nativeFormula,
    KernelArithmetic.FPWord,KernelArithmetic.instInhabitedFPWord]
#axiomsClean kernel_fp_profile
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPEquiv
