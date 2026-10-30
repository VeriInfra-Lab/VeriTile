import bench.repairs.tritonbench_g.fp4_to_bf16.Kernels
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPOperations

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue

/- These internal proof views are definitionally checked against the actual
pasted body. Generic integer operator lemmas remain shared. -/
def exponentCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block*2] "__veritile_helper_9_local_exp_biased_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "__veritile_helper_9_local_exp_biased_f32") 23))]
def normalCode (p : Parameters) : List ComputeStmt :=
  [.assign .nat [p.block*2] "__veritile_helper_10_local_mantissa_f4"
    (.alg (.bitAnd .scalarR (.ref .nat [p.block*2] "__veritile_helper_5_local_x_pos") (.constNat 1))),
   .assign .int [p.block*2] "__veritile_helper_11_local_mantissa_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "__veritile_helper_10_local_mantissa_f4") 22)),
   .assign .int [p.block*2] "__veritile_helper_12_local_output" (.alg (.ref .int [p.block*2] "__veritile_helper_11_local_mantissa_f32")),
   .assign .int [p.block*2] "__veritile_helper_13_local_result"
    (.alg (orExpr (.ref .int [p.block*2] "__veritile_helper_9_local_exp_biased_f32") (.ref .int [p.block*2] "__veritile_helper_11_local_mantissa_f32"))),
   .assign .int [p.block*2] "__veritile_helper_13_local_result" (.alg (.where (.ref .bool [p.block*2] "__veritile_helper_6_local_zero_mask")
     (.broadcast (.constInt 0) [p.block*2]) (.ref .int [p.block*2] "__veritile_helper_13_local_result")))]
def signCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block*2] "__veritile_helper_13_local_result" (.alg (.where (.ref .bool [p.block*2] "__veritile_helper_7_local_denormal_mask")
     (.broadcast (.constInt 1056964608) [p.block*2]) (.ref .int [p.block*2] "__veritile_helper_13_local_result"))),
   .assign .int [p.block*2] "__veritile_helper_14_local_sign_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "__veritile_helper_4_local_sign_f4") 28)),
   .assign .int [p.block*2] "__veritile_helper_13_local_result"
    (.alg (orExpr (.ref .int [p.block*2] "__veritile_helper_13_local_result") (.ref .int [p.block*2] "__veritile_helper_14_local_sign_f32")))]

theorem exponentCode_source (p : Parameters) :
    ((Kernels.source p).surfaceBody.drop 17 |>.take 1) = exponentCode p := rfl
theorem normalCode_source (p : Parameters) :
    ((Kernels.source p).surfaceBody.drop 18 |>.take 5) = normalCode p := rfl
theorem signCode_source (p : Parameters) :
    ((Kernels.source p).surfaceBody.drop 23 |>.take 3) = signCode p := rfl

#axiomsClean exponentCode_source
#axiomsClean normalCode_source
#axiomsClean signCode_source
end VeriTile.Bench.Repairs.Fp4ToBf16.Source
