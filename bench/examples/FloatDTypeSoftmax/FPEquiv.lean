import bench.examples.FloatDTypeSoftmax.IO
import bench.examples.FloatDTypeSoftmax.Contract
import bench.examples.FloatDTypeSoftmax.Kernels
import bench.examples.FloatDTypeSoftmax.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Reciprocal
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.FloatDTypeSoftmaxFPEquiv
open VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton
open FP.Structural FP.Guarded

set_option maxHeartbeats 1600000

open scoped VeriTile.Spec

def originalIO (B : Nat) : FP.PrefixKernel₁ where
  io := typedIO B
  domain := domain B
  defaultPrecision := none -- Keep the source's fp32 IO and fp64 work precision.

def reciprocalIO (B : Nat) : FP.PrefixKernel₁ :=
  { originalIO B with io := (originalIO B).io.withKernel (reciprocalKernel "x" "y" B) }

specification softmax_reciprocal_equiv (R : Rules) (B : Nat) (hB : 0 < B) :
    originalIO B ≡[R] reciprocalIO B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := originalIO B) (rhs := reciprocalIO B)
    (structural := fun _ _ => False) rfl rfl
  exact Execution.equivalent R B hB

#print_fp_assumptions softmax_reciprocal_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeSoftmaxFPEquiv
