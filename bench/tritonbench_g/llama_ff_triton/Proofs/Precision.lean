import bench.tritonbench_g.llama_ff_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.Precision
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRangeDyn _ _ _ _ body => body
  | _ => []

def dotAccum (BM BN BK : Nat) (acc weight : RegName) : ComputeStmt :=
  .assign .real [BM, BN] acc (.compute (.alg .fp32
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] acc)
      (.castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
        (.ref .fp16 [BM, BK] "a") (.ref .fp16 [BK, BN] weight))))))

/-- The input conversion and native integer power run at fp32; RMS-weight
multiplication remains fp16. Both independent dot operations take fp16 inputs,
produce fp32, then add to separate fp32 accumulators. The tail stays fp32 and
the destination conversion is fp16. The source's cdiv expression is retained. -/
theorem computation (fp8 variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) :
    let body := (TypedSource.kernel fp8 variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).surfaceBody
    let loop := loopBody body[13]!
    body[13]! = .forRangeDyn "_i" (.constNat 0) (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK)) (.constNat 1) loop ∧
    loop[1]! = .assign .real [BM, BK] "a_sum" (.compute (.alg .fp32
      (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BK] "a_sum")
        (.libdevicePowi .scalarR (.castFloat .fp16 .real (.ref .fp16 [BM, BK] "a")) (.constInt 2))))) ∧
    loop[if fp8 then 5 else 3]! = .assign .fp16 [BM, BK] "a" (.alg
      (.mul .fp16 (.consR (.consSame .nil)) (.ref .fp16 [BM, BK] "a") (.ref .fp16 [1, BK] "rms_w"))) ∧
    loop[if fp8 then 10 else 5]! = dotAccum BM BN BK "acc1" "b" ∧
    loop[if fp8 then 15 else 7]! = dotAccum BM BN BK "acc2" "c" ∧
    body[14]! = .assign .real [BM] "a_mean" (.compute (.alg .fp32
      (.add .real .scalarR (.div .real .scalarR
        (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BK] "a_sum"))
        (.const (K : ℝ))) (.const EPS)))) ∧
    body[15]! = .assign .real [BM] "a_norm" (.compute (.alg .fp32 (.rsqrt (.ref .real [BM] "a_mean")))) ∧
    body[23]! = .store .fp16 [BM, BN] (.ptr (.ref .ptr [BM, BN] "out_ptrs"))
      (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [BM, BN] "accumulator"))))
      (.mask (.ref .bool [BM, BN] "out_mask")) := by
  cases fp8 <;> cases variant <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The three signed-byte loads preserve their exact reinterpretations and
all subsequent conversions; weights B/C have an extra fp32 conversion that
RMS weights do not. None of these operations is a numerical int-to-float cast. -/
theorem byte_conversions (variant : Bool) (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK : Nat) (EPS : ℝ) :
    let loop := loopBody ((TypedSource.kernel Bool.true variant A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK EPS).surfaceBody[13]!)
    loop[3]! = .assign .f8e5 [1, BK] "rms_w" (.alg (.decodeFloat8E5 .f8e5 (.ref .int [1, BK] "rms_w"))) ∧
    loop[4]! = .assign .fp16 [1, BK] "rms_w" (.alg (.castFloat .f8e5 .fp16 (.withPrecision .f8e5 .fp32 (.ref .f8e5 [1, BK] "rms_w")))) ∧
    loop[7]! = .assign .f8e5 [BK, BN] "b" (.alg (.decodeFloat8E5 .f8e5 (.ref .int [BK, BN] "b"))) ∧
    loop[8]! = .assign .real [BK, BN] "b" (.compute (.alg .fp32 (.castFloat .f8e5 .real (.ref .f8e5 [BK, BN] "b")))) ∧
    loop[9]! = .assign .fp16 [BK, BN] "b" (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [BK, BN] "b")))) ∧
    loop[12]! = .assign .f8e5 [BK, BN] "c" (.alg (.decodeFloat8E5 .f8e5 (.ref .int [BK, BN] "c"))) ∧
    loop[13]! = .assign .real [BK, BN] "c" (.compute (.alg .fp32 (.castFloat .f8e5 .real (.ref .f8e5 [BK, BN] "c")))) ∧
    loop[14]! = .assign .fp16 [BK, BN] "c" (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [BK, BN] "c")))) := by
  cases variant <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.Precision
