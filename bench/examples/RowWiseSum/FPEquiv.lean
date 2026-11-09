import bench.examples.RowWiseSum.IO
import bench.examples.RowWiseSum.Contract
import bench.examples.RowWiseSum.Kernels
import bench.examples.RowWiseSum.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ReductionSchedule
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fin.Rev

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.RowWiseSumFPEquiv
open VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton
open FP.Structural FP.Equational
open FP.Guarded FP.ReductionSchedule FP.ScalarReduction

set_option maxHeartbeats 1600000

open scoped VeriTile.Spec

def original (nCol B : Nat) : FP.Kernel₁ where
  io := originalIO nCol B
  profile := FP.fp32
  domain := domain nCol B
  comparisons := []

def reversed (nCol B : Nat) : FP.Kernel₁ :=
  { original nCol B with io := reversedIO nCol B }

/-- Reversal uses strictly certified scalar commutation together with the
admitted association and zero rules used by tree normalization. Every
intermediate guard, successful execution and both memory frames are retained. -/
specification rowwise_sum_equiv (nCol B : Nat) (R : Rules) :
    original nCol B ≡[R] reversed nCol B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := original nCol B) (rhs := reversed nCol B)
    (structural := fun _ _ => False) rfl rfl
  apply FP.Kernel₁.ofExecution
  exact Execution.equivalent nCol B R

#print_fp_assumptions rowwise_sum_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.RowWiseSumFPEquiv
