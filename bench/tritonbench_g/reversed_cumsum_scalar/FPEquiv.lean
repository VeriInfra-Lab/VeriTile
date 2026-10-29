import bench.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.FP
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedLoop
import VeriTile.Triton.Float.ProfiledRewrite

/-! Ordinary public profiled floating equivalence for the complete typed
source and candidate. No numerical law or floating domain guard is needed:
the candidate changes only natural index multiplication. All precision,
scan, reduction, padding, casts, stores and failures remain in context. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : Program where
  kernel := Kernels.original dtype out_dtype s o T BT
  domain := []

def indexCommuted (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : Program where
  kernel := Kernels.indexCommuted dtype out_dtype s o T BT
  domain := []

specification reversed_cumsum_scalar_equiv (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    original dtype out_dtype s o T BT ≡[assumptions] indexCommuted dtype out_dtype s o T BT := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original dtype out_dtype s o T BT) (rhs := indexCommuted dtype out_dtype s o T BT)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D _ u _
  exact FPProof.whole_state dtype out_dtype s o T BT (engine M) u

/-- Successful complete executions of these same public programs under the
same profiled engine, with empty registers, bf16 input and fp16 output. -/
theorem nonempty_pair :
    ∃ v, FP.Structural.exec (engine TypedExecution.model)
      (original .bf16 .fp16 "s" "o" 17 16).kernel TypedExecution.initial = some v ∧
      FP.Structural.exec (engine TypedExecution.model)
        (indexCommuted .bf16 .fp16 "s" "o" 17 16).kernel TypedExecution.initial = some v :=
  TypedExecution.nonempty_pair

#axiomsClean reversed_cumsum_scalar_equiv
#axiomsClean nonempty_pair

end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Equivalence
