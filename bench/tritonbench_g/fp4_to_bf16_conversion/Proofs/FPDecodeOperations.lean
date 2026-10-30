import bench.tritonbench_g.fp4_to_bf16.Proofs.FPOperations
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def exponentCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block*2] "exp_biased_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "exp_biased_f32") 23))]
def normalCode (p : Parameters) : List ComputeStmt :=
  [.assign .nat [p.block*2] "mantissa_f4"
    (.alg (.bitAnd .scalarR (.ref .nat [p.block*2] "x_pos") (.constNat 1))),
   .assign .int [p.block*2] "mantissa_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "mantissa_f4") 22)),
   .assign .int [p.block*2] "output" (.alg (.ref .int [p.block*2] "mantissa_f32")),
   .assign .int [p.block*2] "result"
    (.alg (orExpr (.ref .int [p.block*2] "exp_biased_f32") (.ref .int [p.block*2] "mantissa_f32"))),
   .assign .int [p.block*2] "result" (.alg (.where (.ref .bool [p.block*2] "zero_mask")
     (.broadcast (intConst 0) [p.block*2]) (.ref .int [p.block*2] "result")))]
def signCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block*2] "result" (.alg (.where (.ref .bool [p.block*2] "denormal_mask")
     (.broadcast (intConst 1056964608) [p.block*2]) (.ref .int [p.block*2] "result"))),
   .assign .int [p.block*2] "sign_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "sign_f4") 28)),
   .assign .int [p.block*2] "result"
    (.alg (orExpr (.ref .int [p.block*2] "result") (.ref .int [p.block*2] "sign_f32")))]

theorem exponentCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 17 |>.take 1) = exponentCode p := rfl
theorem normalCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 18 |>.take 5) = normalCode p := rfl
theorem signCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 23 |>.take 3) = signCode p := rfl
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
