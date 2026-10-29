import bench.tritonbench_g.reversed_cumsum.Kernels
import bench.tritonbench_g.reversed_cumsum.Proofs.FP
import bench.tritonbench_g.reversed_cumsum.Proofs.TypedLoop
import VeriTile.Triton.Float.ProfiledRewrite

/-! Complete typed state equivalence with no arithmetic assumptions. Only
the operands of the discrete triangular comparison are reversed. Independent
real correctness is in Correct; typed success uses the explicit mathematical
consistency algebra below and makes no IEEE/native execution claim. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (dtype out_dtype : FloatDType) (s o : RegionName)
    (sh st sd T S BT BS : Nat) : Program where
  kernel := TypedSource.original dtype out_dtype s o sh st sd T S BT BS
  domain := []

def candidate (dtype out_dtype : FloatDType) (s o : RegionName)
    (sh st sd T S BT BS : Nat) : Program where
  kernel := TypedSource.compareRight dtype out_dtype s o sh st sd T S BT BS
  domain := []

specification reversed_cumsum_compare_right_equiv (dtype out_dtype : FloatDType)
    (s o : RegionName) (sh st sd T S BT BS : Nat) :
    original dtype out_dtype s o sh st sd T S BT BS ≡[assumptions]
      candidate dtype out_dtype s o sh st sd T S BT BS := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original dtype out_dtype s o sh st sd T S BT BS)
    (rhs := candidate dtype out_dtype s o sh st sd T S BT BS)
    (structural := fun _ _ => False) rfl (by rfl)
  intro α _ A D _ u _
  exact FPProof.whole_state dtype out_dtype s o sh st sd T S BT BS (engine A) u

/-- With independent bf16 input and fp16 output, every layout and initial
state has a successful complete run in the stated consistency backend. -/
theorem reversed_cumsum_typed_nonempty (p : RealProgram.Layout) (s o : RegionName)
    (u : FP.Structural.State (WithBot ℝ)) :
    ∃ final, FP.Structural.exec (engine TypedExecution.model)
      (original .bf16 .fp16 s o p.headStride p.timeStride p.channelStride
        p.T p.S p.BT p.BS).kernel u = some final ∧
      FP.Structural.exec (engine TypedExecution.model)
        (candidate .bf16 .fp16 s o p.headStride p.timeStride p.channelStride
          p.T p.S p.BT p.BS).kernel u = some final :=
  TypedExecution.nonempty_pair p s o u

/-- Empty registers, nonzero program IDs, two chunks, both boundary tails,
independent storage casts and a nonzero unspecified-value oracle. -/
theorem reversed_cumsum_partial_tail_witness :
    ∃ final, FP.Structural.exec (engine TypedExecution.model)
      (original .bf16 .fp16 "s" "o" 561 33 1 17 33 16 32).kernel TypedExecution.initial = some final ∧
      FP.Structural.exec (engine TypedExecution.model)
        (candidate .bf16 .fp16 "s" "o" 561 33 1 17 33 16 32).kernel TypedExecution.initial = some final :=
  reversed_cumsum_typed_nonempty ⟨561,33,1,17,33,16,32⟩ "s" "o" TypedExecution.initial

end VeriTile.Bench.TritonBenchG.ReversedCumsum.Equivalence
