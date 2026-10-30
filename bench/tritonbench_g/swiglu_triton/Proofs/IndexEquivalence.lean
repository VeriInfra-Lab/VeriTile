import bench.tritonbench_g.swiglu_triton.IO
import bench.tritonbench_g.swiglu_triton.Proofs.Mathematical_Execution
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Mathematical_FP

namespace VeriTile.Bench.Repairs.SwigluTriton.PairProof
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluTriton
open VeriTile.Bench.TritonBenchG.GegluTanhTriton.FPProof
open scoped VeriTile.Spec
noncomputable section
set_option maxRecDepth 10000

/-- The source prefix proof already covers signed integer multiplication;
the remaining statements are definitionally identical, including the FMA. -/
theorem equivalent (backward : Bool) (block cols : Nat) :
    Pair.original backward block cols ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      Pair.indexCommuted backward block cols := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases backward <;> rfl)
  intro α _ M D _ s _
  cases backward
  · change run (engine M)
        (typedFwd .fp32 .fp32 Bool.false "A" "B" "C" cols cols block).surfaceBody s =
      run (engine M)
        (typedFwd .fp32 .fp32 Bool.true "A" "B" "C" cols cols block).surfaceBody s
    change run (engine M)
        (sourcePrefix Bool.false "A" "B" "C" "a_ptr" "b_ptr" "c_ptr" cols ++
          (typedFwd .fp32 .fp32 Bool.false "A" "B" "C" cols cols block).surfaceBody.drop 4) s =
      run (engine M)
        (sourcePrefix Bool.true "A" "B" "C" "a_ptr" "b_ptr" "c_ptr" cols ++
          (typedFwd .fp32 .fp32 Bool.false "A" "B" "C" cols cols block).surfaceBody.drop 4) s
    exact prefix_run M s "A" "B" "C" "a_ptr" "b_ptr" "c_ptr" cols _
  · change run (engine M)
        (typedBwd .fp32 .fp32 .fp32 Bool.false "DC" "A" "B" cols cols block).surfaceBody s =
      run (engine M)
        (typedBwd .fp32 .fp32 .fp32 Bool.true "DC" "A" "B" cols cols block).surfaceBody s
    change run (engine M)
        (sourcePrefix Bool.false "DC" "A" "B" "dc_ptr" "a_ptr" "b_ptr" cols ++
          (typedBwd .fp32 .fp32 .fp32 Bool.false "DC" "A" "B" cols cols block).surfaceBody.drop 4) s =
      run (engine M)
        (sourcePrefix Bool.true "DC" "A" "B" "dc_ptr" "a_ptr" "b_ptr" cols ++
          (typedBwd .fp32 .fp32 .fp32 Bool.false "DC" "A" "B" cols cols block).surfaceBody.drop 4) s
    exact prefix_run M s "DC" "A" "B" "dc_ptr" "a_ptr" "b_ptr" cols _

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.SwigluTriton.PairProof
