import bench.tritonbench_g.rms_rbe_matmul.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FPEquiv
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input : Storage) (fp8 variant : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta eps : ℝ) : Program where
  kernel := kernel input fp8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
    sob som son BM BN BK start rbe theta eps
  domain := []

/-- Full typed state and failure equivalence for the exact output-index
commutation, for arbitrary arithmetic, integer-power, dot and decode backends.
All E5M2 payloads remain in scope; no success or finiteness premise is assumed.
This formal floating-input family is not a native frontend admission result. -/
specification source_equivalence (input : Storage) (fp8 : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta eps : ℝ) :
    program input fp8 Bool.false X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta eps ≡[assumptions]
    program input fp8 Bool.true X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta eps := by
  cases input <;> cases fp8 <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.FPEquiv
