import bench.tritonbench_g.rms_matmul_rbe.Kernels
import bench.tritonbench_g.rms_matmul_rbe.Proofs.LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Memory.TraceRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.RealSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- The exact source ceiling-division expression, including natural division
at zero. Native launch validity is a separate domain condition. -/
def ceilDivOp (K BK : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K)
    (.constNat BK)) (.constNat 1)) (.constNat BK)

/-- Retain raw signed-byte loads, decoder failure and every conversion
assignment in the weight arm. No nonfinite payload receives a default value. -/
def weightLoad (fp8 : Bool) (st : Stmt) : List Stmt :=
  match st with
  | .assign .real sh name (.load .real (.ptr ptr) .none) =>
      if fp8 && (name == "rms_w" || name == "w") then
        [.assign .int sh name (.load .int (.ptr ptr) .none),
         .assign .real sh name (.decodeFloat8E5 .real (.ref .int sh name)),
         .assign .real sh name (.ref .real sh name)] ++
          (if name == "rms_w" then [] else [.assign .real sh name (.ref .real sh name)])
      else [st]
  | _ => [st]

def loop (fp8 : Bool) (BM BN BK sxk swk srms : Nat) : List Stmt :=
  ((rrmLoopBody BM BN BK sxk swk srms).map Stmt.eraseDType).flatMap (weightLoad fp8)

/-- Both output storage types erase to the same real store. The source
candidate changes only the natural row-index product at tail position three. -/
def tail (variant : Bool) (OUT : RegionName) (M N BM BN BK K sob som son : Nat)
    (EPS : ℝ) : List Stmt :=
  let old := (rrmStoreTail OUT M N BM BN BK K sob som son EPS).map Stmt.eraseDType
  if variant then old.take 3 ++
    [.assign .nat [BM] "offs_m" (.add .nat .scalarL
      (.mul .nat .nil (.constNat BM) (.ref .nat [] "pid_m")) (.arange BM))] ++ old.drop 4
  else old

def body (fp8 variant : Bool) (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat) (EPS : ℝ) : List Stmt :=
  (rrmPrologue X W RMS M N BM BN BK sxb sxm sxk swk swn srms).map Stmt.eraseDType ++
    .forRangeDyn "_i" (.constNat 0) (ceilDivOp K BK) (.constNat 1)
      (loop fp8 BM BN BK sxk swk srms) ::
      tail variant OUT M N BM BN BK K sob som son EPS

/-- Complete all-profile projection, including actual K ceiling division,
the FP8 arm and every output statement. This theorem erases floating dtypes
only; it does not erase byte decoding or truncate the unmasked tail. -/
theorem projection (input : TypedSource.Storage) (fp8 outputFP8 variant : Bool)
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ) :
    (TypedSource.kernel input fp8 outputFP8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms
      sob som son BM BN BK start rbe theta EPS).eraseDType.toAlgorithm? =
      Except.ok ⟨[X, W, RMS], [OUT],
        body fp8 variant X W RMS OUT M N K sxb sxm sxk swk swn srms sob som son BM BN BK EPS⟩ := by
  cases input <;> cases fp8 <;> cases outputFP8 <;> cases variant <;>
    simp only [TypedSource.kernel,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
      MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
      List.map_cons, List.map_nil]
  all_goals
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    simp [body, loop, tail, weightLoad, rrmPrologue, rrmLoopBody, rrmStoreTail, ceilDivOp,
      Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, FloatDType.toTileDType, VeriTile.Triton.eraseDType,
      TileShape.eraseAxis]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.RealSource
