import bench.tritonbench_g.llama_ff_triton.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def program (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) : Program where
  kernel := TypedSource.kernel fp8 variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS
  domain := []

/-- Complete state and failure equivalence, including both native-power and
E5M2 backend calls, all casts, two dots, the actual cdiv loop and masked output.
Only the natural row-index product commutes. No floating assumption is used;
all signed int8 payloads, including nonfinite E5M2 encodings, remain in scope. -/
specification source_equivalence (fp8 : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) :
    program fp8 Bool.false A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS ≡[assumptions]
      program fp8 Bool.true A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS := by
  cases fp8 <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.FPEquiv
