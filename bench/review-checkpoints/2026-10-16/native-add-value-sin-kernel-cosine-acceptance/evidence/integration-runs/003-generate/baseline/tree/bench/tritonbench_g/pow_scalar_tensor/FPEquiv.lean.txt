import bench.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerKernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerPrecision
import bench.tritonbench_g.pow_scalar_tensor.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv
open VeriTile.Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) : Program where
  kernel := TypedSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B
  domain := []

/-- Full typed state/failure equivalence for the complete fp32-exponent source,
both constexpr branches and every formal floating destination. Only the
natural tile-offset product changes, so no numerical FP assumptions apply.
Signed strides and the separate native int32-exponent path are not covered. -/
specification floating_source_equivalence (output : TypedSource.FloatingOutput)
    (oneTile : Bool) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    program output oneTile Bool.false base X Y sx sy N tasks tiles B ≡[assumptions]
      program output oneTile Bool.true base X Y sx sy N tasks tiles B := by
  cases output <;> cases oneTile <;> equiv_decompose <;> fp_prove

def integerProgram (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) : Program where
  kernel := TypedIntegerSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B
  domain := []

/-- Complete state/failure equality for the native int32 exponent source and
all six floating destinations. The original integer values, range checks and
native powi callback are retained, including missing-backend failure. Only the
natural offset product changes; no numerical FP assumption is required. -/
specification integer_source_equivalence (output : TypedSource.FloatingOutput)
    (oneTile : Bool) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    integerProgram output oneTile Bool.false base X Y sx sy N tasks tiles B ≡[assumptions]
      integerProgram output oneTile Bool.true base X Y sx sy N tasks tiles B := by
  cases output <;> cases oneTile <;> equiv_decompose <;> fp_prove

#print_fp_assumptions floating_source_equivalence
#print_fp_assumptions integer_source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv
