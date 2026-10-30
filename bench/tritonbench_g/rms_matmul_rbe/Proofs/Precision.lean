import bench.tritonbench_g.rms_matmul_rbe.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.Precision
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRangeDyn _ _ _ _ body => body
  | _ => []

def inputFloat : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 | .fp64 => .real
  | .f8e4 => .f8e4 | .f8e5 => .f8e5

def xLoad (input : Storage) (BM BK : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [BM, BK] "x" (.compute
      (.load .fp32 (.ptr (.ref .ptr [BM, BK] "x_ptrs")) .none))
  | .fp64 => .assign .real [BM, BK] "x" (.compute (.alg .fp64
      (.load .real (.ptr (.ref .ptr [BM, BK] "x_ptrs")) .none)))
  | other => .assign (inputFloat other).toTileDType [BM, BK] "x" (.alg
      (.load (inputFloat other).toTileDType (.ptr (.ref .ptr [BM, BK] "x_ptrs")) .none))

def xRef (input : Storage) (BM BK : Nat) : Op (inputFloat input).toTileDType [BM, BK] :=
  if input = .fp64 then .withPrecision (inputFloat input) .fp64
    (.ref (inputFloat input).toTileDType [BM, BK] "x")
  else .ref (inputFloat input).toTileDType [BM, BK] "x"

def moment (input : Storage) (BM BK : Nat) : ComputeStmt :=
  .assign .real [BM, BK] "x_sum" (.compute (.alg .fp32
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BK] "x_sum")
      (.libdevicePowi .scalarR (.castFloat (inputFloat input) .real (xRef input BM BK)) (.constInt 2)))))

def halfMultiply (source : FloatDType) (same : Bool) (BM BK : Nat) : ComputeStmt :=
  .assign .fp16 [BM, BK] "x" (.alg (.mul .fp16 (.consR (.consSame .nil))
    (if same then .ref .fp16 [BM, BK] "x"
     else .withPrecision .fp16 .fp32 (.castFloat source .fp16
       (.ref source.toTileDType [BM, BK] "x")))
    (if same then .ref .fp16 [1, BK] "rms_w"
     else .withPrecision .fp16 .fp32 (.ref .fp16 [1, BK] "rms_w"))))

def multiply (input : Storage) (BM BK : Nat) : ComputeStmt :=
  match input with
  | .fp16 => halfMultiply .fp16 Bool.true BM BK
  | .bf16 => halfMultiply .bf16 Bool.false BM BK
  | .f8e4 => halfMultiply .f8e4 Bool.false BM BK
  | .f8e5 => halfMultiply .f8e5 Bool.false BM BK
  | .fp32 => .assign .real [BM, BK] "x" (.compute (.alg .fp32
      (.mul .real (.consR (.consSame .nil)) (.ref .real [BM, BK] "x")
        (.castFloat .fp16 .real (.ref .fp16 [1, BK] "rms_w")))))
  | .fp64 => .assign .real [BM, BK] "x" (.compute (.alg .fp64
      (.mul .real (.consR (.consSame .nil)) (.ref .real [BM, BK] "x")
        (.castFloat .fp16 .real (.withPrecision .fp16 .fp32 (.ref .fp16 [1, BK] "rms_w"))))))

def dotInput : Storage → FloatDType
  | .fp32 | .fp64 => .real
  | _ => .fp16

def accumulation (input : Storage) (BM BN BK : Nat) : ComputeStmt :=
  .assign .real [BM, BN] "accumulator" (.compute (.alg .fp32
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "accumulator")
      (.castFloat .fp32 .real (.dotTyped (dotInput input) .fp16 .fp32 (batch := [])
        (.ref (dotInput input).toTileDType [BM, BK] "x") (.ref .fp16 [BK, BN] "w"))))))

/-- Independent templates for raw byte load, reinterpretation and each
conversion. W has the extra fp32 conversion that RMS does not. -/
def weight (fp8 extra : Bool) (shape : TileShape) (name ptr : RegName) : List ComputeStmt :=
  if fp8 then
    [.assign .int shape name (.alg (.load .int (.ptr (.ref .ptr shape ptr)) .none)),
     .assign .f8e5 shape name (.alg (.decodeFloat8E5 .f8e5 (.ref .int shape name)))] ++
    (if extra then
      [.assign .real shape name (.compute (.alg .fp32 (.castFloat .f8e5 .real (.ref .f8e5 shape name)))),
       .assign .fp16 shape name (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real shape name))))]
     else [.assign .fp16 shape name (.alg (.castFloat .f8e5 .fp16 (.withPrecision .f8e5 .fp32 (.ref .f8e5 shape name))))])
  else [.assign .fp16 shape name (.alg (.load .fp16 (.ptr (.ref .ptr shape ptr)) .none))]

def advance (M N : Nat) (name : RegName) (BK stride : Nat) : ComputeStmt :=
  .assign .ptr [M, N] name (.alg (.ptrAdd .scalarR (.ref .ptr [M, N] name)
    (.mul .nat .nil (.constNat BK) (.constNat stride))))

/-- Every loop statement is independently prescribed, including pointers. -/
def expectedLoop (input : Storage) (fp8 : Bool) (BM BN BK sxk swk srms : Nat) : List ComputeStmt :=
  [xLoad input BM BK, moment input BM BK] ++ weight fp8 Bool.false [1, BK] "rms_w" "rms_w_ptrs" ++
  [multiply input BM BK] ++ weight fp8 Bool.true [BK, BN] "w" "w_ptrs" ++
  [accumulation input BM BN BK, advance BM BK "x_ptrs" BK sxk,
   advance BK BN "w_ptrs" BK swk, advance 1 BK "rms_w_ptrs" BK srms]

def mean (BM BK K : Nat) (EPS : ℝ) : ComputeStmt :=
  .assign .real [BM] "x_mean" (.compute (.alg .fp32
    (.add .real .scalarR (.div .real .scalarR
      (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BK] "x_sum"))
      (.const (K : ℝ))) (.const EPS))))

def norm (BM : Nat) : ComputeStmt :=
  .assign .real [BM] "x_norm" (.compute (.alg .fp32 (.rsqrt (.ref .real [BM] "x_mean"))))

def scale (BM BN : Nat) : ComputeStmt :=
  .assign .real [BM, BN] "accumulator" (.compute (.alg .fp32
    (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BN] "accumulator")
      (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "x_norm")))))

def output (outputFP8 : Bool) (BM BN : Nat) : ComputeStmt :=
  let d : FloatDType := if outputFP8 then .f8e5 else .fp16
  .store d.toTileDType [BM, BN] (.ptr (.ref .ptr [BM, BN] "out_ptrs"))
    (.alg (.castFloat .real d (.withPrecision .real .fp32 (.ref .real [BM, BN] "accumulator"))))
    (.mask (.ref .bool [BM, BN] "out_mask"))

/-- All 48 callee profiles preserve the complete independent precision and
conversion template. The fp64 X-local multiplication remains fp64 while the
source dot result and separate carry are fp32. Formal typing does not prove
native frontend admission of mixed dot operands. -/
theorem profiles (input : Storage) (fp8 outputFP8 variant : Bool)
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son BM BN BK : Nat)
    (start : Int) (rbe : Bool) (theta EPS : ℝ) :
    let source := (kernel input fp8 outputFP8 variant X W RMS OUT M N K
      sxb sxm sxk swk swn srms sob som son BM BN BK start rbe theta EPS).surfaceBody
    source.length = 21 ∧
    source[12]! = .forRangeDyn "_i" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK))
      (.constNat 1) (expectedLoop input fp8 BM BN BK sxk swk srms) ∧
    source[13]! = mean BM BK K EPS ∧ source[14]! = norm BM ∧
    source[15]! = scale BM BN ∧ source[20]! = output outputFP8 BM BN := by
  cases input <;> cases fp8 <;> cases outputFP8 <;> cases variant <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Q remains fp16 in both weight arms; K/V are E5M2 precisely in the byte
arm. All three independent dot loops retain their own K strides and decoding. -/
theorem qkv_profiles (input : Storage) (fp8 variant : Bool)
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK : Nat)
    (start : Int) (theta EPS : ℝ) :
    let source := (qkv input fp8 variant X QW KW VW RMS Q KOut V M N K
      sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK start theta EPS).surfaceBody
    source.length = 63 ∧
    loopBody source[12]! = expectedLoop input fp8 BM BN BK sxk sqwk srms ∧
    loopBody source[33]! = expectedLoop input fp8 BM BN BK sxk skwk srms ∧
    loopBody source[54]! = expectedLoop input fp8 BM BN BK sxk svwk srms ∧
    source[20]! = output Bool.false BM BN ∧ source[41]! = output fp8 BM BN ∧ source[62]! = output fp8 BM BN := by
  cases input <;> cases fp8 <;> cases variant <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.Precision
