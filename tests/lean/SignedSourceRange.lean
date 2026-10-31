import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SignedSourceRange
open VeriTile.Triton

def source (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    for i in range(tl.cast(0, tl.int64), tl.cast(N, tl.int64), tl.cast(1, tl.int64)):
        last = i
  }

/-- All three source casts remain in the complete loop statement, including
its explicitly signed step before the positive-stride range conversion. -/
theorem complete_loop (N : Nat) :
    (source N).surfaceBody=
      [.forRangeInt "i" (.castNatToInt (.constNat 0)) (.castNatToInt (.constNat N))
        (.castIntToNat (.castNatToInt (.constNat 1)))
        [.assign .int [] "last" (.alg (.ref .int [] "i"))]] := by rfl

def integerConstant : ComputeKernel :=
  with_triton_signature ({} : CallSignature) in triton {
    x = tl.cast(1, tl.int64)
  }

def floatingConstant : ComputeKernel :=
  with_triton_signature ({} : CallSignature) in triton {
    x = tl.cast(1.0, tl.int64)
  }

theorem integer_literal : integerConstant.surfaceBody=
    [.assign .int [] "x" (.alg (.castNatToInt (.constNat 1)))] := by rfl

theorem floating_literal : floatingConstant.surfaceBody=
    [.assign .int [] "x" (.alg (.floatToInt none .real 64 (.const 1.0)))] := by rfl

def integerExpressions : ComputeKernel :=
  with_triton_signature ({scalars := [("offset",.integer)]} : CallSignature) in triton {
    last = offset - 1
    safe = tl.where(offset > 0, offset, 0)
  }

theorem integer_arithmetic : integerExpressions.surfaceBody=
    [.assign .int [] "last" (.alg (.sub .int .nil (.ref .int [] "offset") (.constInt 1))),
     .assign .int [] "safe" (.alg (.where
       (.gt .int .nil (.ref .int [] "offset") (.constInt 0))
       (.ref .int [] "offset") (.constInt 0)))] := by rfl

/-- A real-valued caller argument must establish floating arithmetic before
the integer spelling of a literal can select an integer carrier. -/
def floatingAntiquote (p : ℝ) : ComputeKernel := triton {
  remaining = 1 - $(p)
}

theorem floating_antiquote (p : ℝ) : (floatingAntiquote p).surfaceBody=
    [.assign .real [] "remaining" (.alg (.sub .real .nil (.const 1) (.const p)))] := by rfl

/-- The source cast supplies an integer indicator to quotient/remainder ceiling. -/
def quotientRemainderCeiling (B : Nat) : ComputeKernel :=
  with_triton_signature ({scalars := [("M",.integer)],constexpr := ["B"]} : CallSignature) in triton {
    q = M // B + (M % B != 0).to(tl.int64)
  }

theorem quotient_remainder_ceiling (B : Nat) : (quotientRemainderCeiling B).surfaceBody=
    [.assign .int [] "q" (.alg (.add .int .nil
      (.tritonDiv .int .nil (.ref .int [] "M") (.castNatToInt (.constNat B)))
      (.boolToInt (.ne .int .nil
        (.tritonMod .int .nil (.ref .int [] "M") (.castNatToInt (.constNat B))) (.constInt 0)))))] := by rfl

#axiomsClean quotient_remainder_ceiling
/-- Address inference must retain the source's signed vector cast. -/
def signedPointerLanes (A B : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("A",.fp16),("B",.fp16)]} : CallSignature) in triton {
    offsets = tl.arange(0, 4).to(tl.int64)
    x = tl.load(A + offsets)
    tl.store(B + offsets, x)
  }

theorem signed_pointer_lanes (A B : RegionName) : (signedPointerLanes A B).surfaceBody[0]?=
    some (.assign .int [4] "offsets" (.alg (.castNatToInt (.arange 4)))) := by rfl


def signedCeiling : ComputeKernel :=
  with_triton_signature ({scalars := [("N",.integer)]} : CallSignature) in triton {
    blocks = tl.cdiv(N, 8)
  }

theorem signed_ceiling_type : signedCeiling.surfaceBody=
    [.assign .int [] "blocks" (.alg (.tritonDiv .int .nil
      (.add .int .nil (.ref .int [] "N")
        (.sub .int .nil (.constInt 8) (.constInt 1)))
      (.constInt 8)))] := by rfl

/-- Signed operands are not clipped to a natural number before the library body. -/
theorem signed_ceiling_negative {α : Type} [Inhabited α] (A : FP.Structural.Algebra α)
    (s : FP.Structural.State α) :
    (FP.Structural.exec A signedCeiling
      (s.setReg "N" .int [] (fun _=>(-17 : Int)))).map
        (fun t=>t.regs .int [] "blocks")=some (some (fun _=>(-1 : Int))) := by
  simp [signedCeiling,ComputeKernel.surfaceBody,FP.Structural.exec,FP.Structural.run,
    FP.Structural.step,FP.Structural.stepAlg,FP.Structural.evalExpr,FP.Structural.evalOp_unfold,
    FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,
    FP.Structural.integralFloorDiv,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean signed_ceiling_negative

#axiomsClean signed_ceiling_type

#axiomsClean signed_pointer_lanes
#axiomsClean complete_loop
#axiomsClean integer_literal
#axiomsClean floating_literal
#axiomsClean integer_arithmetic
#axiomsClean floating_antiquote
end VeriTile.Tests.SignedSourceRange
