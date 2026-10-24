import bench.tritonbench_g.llama_ff_triton.Kernels
import bench.tritonbench_g.llama_ff_triton.Proofs.LegacyReal
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Memory.TraceRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Source ceil-div arithmetic, including its natural zero-divisor behavior. -/
def ceilDivOp (K BK : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK)

/-- Preserve unmasked signed-byte loads and every erased conversion assignment.
This normalization changes no source operation and keeps decode failure. -/
def weightLoad (fp8 : Bool) (st : Stmt) : List Stmt :=
  match st with
  | .assign .real sh name (.load .real (.ptr ptr) .none) =>
      if fp8 && (name == "rms_w" || name == "b" || name == "c") then
        [.assign .int sh name (.load .int (.ptr ptr) .none),
         .assign .real sh name (.decodeFloat8E5 .real (.ref .int sh name)),
         .assign .real sh name (.ref .real sh name)] ++
          (if name == "rms_w" then [] else [.assign .real sh name (.ref .real sh name)])
      else [st]
  | _ => [st]

/-- The actual loop, with native integer power and either weight storage arm. -/
def loop (fp8 : Bool) (BM BN BK sak sw1k sw3k srms : Nat) : List Stmt :=
  let old := (ffLoopBody BM BN BK sak sw1k sw3k srms).map Stmt.eraseDType
  (old.take 1 ++
    [Stmt.assign .real [BM, BK] "a_sum"
      (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BK] "a_sum")
        (.libdevicePowi .scalarR (.ref .real [BM, BK] "a") (.constInt 2)))] ++
    old.drop 2).flatMap (weightLoad fp8)

/-- The output candidate commutes one natural scalar index product. -/
def tail (variant : Bool) (OUT : RegionName) (M N BM BN BK K soutm soutn : Nat) (EPS : ℝ) : List Stmt :=
  let old := (ffStoreTail OUT M N BM BN BK K soutm soutn EPS).map Stmt.eraseDType
  if variant then old.take 5 ++
    [.assign .nat [BM] "offs_outm" (.add .nat .scalarL
      (.mul .nat .nil (.constNat BM) (.ref .nat [] "pid_m")) (.arange BM))] ++ old.drop 6
  else old

/-- Complete actual source body. The ceiling trip count is computed at runtime
from the dimension parameters and does not require K divisibility. -/
def body (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) : List Stmt :=
  (ffPrologue A W1 W3 RMS M N BM BN BK sam sak sw1k sw1n sw3k sw3n srms).map Stmt.eraseDType ++
    .forRangeDyn "_i" (.constNat 0) (ceilDivOp K BK) (.constNat 1)
      (loop fp8 BM BN BK sak sw1k sw3k srms) ::
      tail variant OUT M N BM BN BK K soutm soutn EPS

theorem projection (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) :
    (TypedSource.kernel fp8 variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).eraseDType.toAlgorithm? =
      Except.ok ⟨[A, W1, W3, RMS], [OUT], body fp8 variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS⟩ := by
  cases fp8 <;> cases variant <;>
    simp only [TypedSource.kernel, TypedSource.original, TypedSource.candidate,
      TypedSource.originalFP8, TypedSource.candidateFP8,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, List.map_cons, List.map_nil]
  all_goals
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    simp [body, loop, tail, weightLoad, ffPrologue, ffLoopBody, ffStoreTail, ceilDivOp,
      Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, FloatDType.toTileDType, VeriTile.Triton.eraseDType,
      TileShape.eraseAxis]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.RealSource
