import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardProfiles
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFP

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def fullProgram (gradient saved destination : Typed.Storage) (commuted : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) : Program where
  kernel := Typed.kernel gradient saved destination commuted GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16
  domain := []

/-- Whole-state equality for the full independently typed formal family.
This is quantified over abstract FP algebras. Native acceptance is narrower:
in particular the original bf16 LOG path and the homogeneous fp8 profiles
are rejected by the separately recorded native lowering matrix. -/
specification backward_full_index_equiv (gradient saved destination : Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    fullProgram gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 ≡[assumptions]
    fullProgram gradient saved destination Bool.true GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 := by
  cases gradient <;> cases saved <;> cases destination <;> equiv_decompose <;> fp_prove

#print_fp_assumptions backward_full_index_equiv
#axiomsClean backward_full_index_equiv
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating
