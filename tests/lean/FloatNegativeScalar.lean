import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Tests.FloatNegativeScalar
open VeriTile.Triton

def halfNegative (T O : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float16, O = tl.float16
  target = tl.load(T)
  res = target * -1
  tl.store(O, res)
}

/-- A negative constexpr is materialized in the half tensor's type without runtime subtraction. -/
example (T O : RegionName) :
    (halfNegative T O).surfaceBody[1]? = some (.assign .fp16 [] "res"
      (.alg (.mul .fp16 .nil (.ref .fp16 [] "target") (.constFloat .fp16 (-1))))) := rfl

/-- The same-storage store retains the narrow value without an unrequested wider carrier. -/
example (T O : RegionName) :
    (halfNegative T O).surfaceBody[2]? = some (.store .fp16 []
      (.region O.cast (.constNat 0)) (.alg (.ref .fp16 [] "res")) .none) := rfl

def bfloatNegative (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.bfloat16
  target = tl.load(T)
  res = (-1) * target
}

/-- Parentheses and left scalar position retain the same weak-scalar rule. -/
example (T : RegionName) :
    (bfloatNegative T).surfaceBody[1]? = some (.assign .bf16 [] "res"
      (.alg (.mul .bf16 .nil (.constFloat .bf16 (-1)) (.ref .bf16 [] "target")))) := rfl

def singleNegative (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float32
  target = tl.load(T)
  res = target * -1
}

/-- Typed fp32 likewise receives one constant, with no additional backend subtraction. -/
example (T : RegionName) :
    (singleNegative T).surfaceBody[1]? = some (.assign .real [] "res"
      (.compute (.alg .fp32 (.mul .real .nil (.ref .real [] "target") (.const (-1)))))) := rfl

def doubleNegative (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float64
  target = tl.load(T)
  res = target * -1
}

/-- Typed fp64 also materializes the constexpr directly in the selected computation type. -/
example (T : RegionName) :
    (doubleNegative T).surfaceBody[1]? = some (.assign .real [] "res"
      (.compute (.alg .fp64 (.mul .real .nil (.ref .real [] "target") (.const (-1)))))) := rfl

def halfRuntimeNegation (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float16
  target = tl.load(T)
  res = -target
}

/-- Runtime tensor negation uses the dedicated sign operation in the tensor's own type.
Unlike subtraction from +0, this preserves the IEEE sign-bit behavior at zero. -/
example (T : RegionName) :
    (halfRuntimeNegation T).surfaceBody[1]? = some (.assign .fp16 [] "res"
      (.alg (.negFloat .fp16 (.ref .fp16 [] "target")))) := rfl

def legacyRealNegative (T : RegionName) : ComputeKernel := triton {
  target = tl.load(T)
  res = target * -1
}

/-- Negative constexpr values are materialized directly in the untyped mathematical carrier too. -/
example (T : RegionName) :
    (legacyRealNegative T).surfaceBody[1]? = some (.assign .real [] "res"
      (.alg (.mul .real .nil (.ref .real [] "target")
        (.const (-1))))) := rfl


def singleRuntimeNegation (T : RegionName) : ComputeKernel := triton {
  tl.region T = tl.float32
  target = tl.load(T)
  res = -target
}

/-- The actual load/negate kernel reaches the fp32 sign-bit implementation for
all stored bit patterns. The arbitrary remainder supplies no negation law. -/
theorem single_runtime_negation
    [Inhabited (FP.Value .fp32)] (rest : FP.Structural.Algebra (FP.Value .fp32))
    (T : RegionName) (s : FP.Structural.State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest)
        (singleRuntimeNegation T) s = some final ∧
      final.regs .real [] "res" = some (fun _ => ((s.mem T 0).read .real).neg) := by
  simp [singleRuntimeNegation, FP.Structural.exec,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.State.setReg, FP.Structural.ofFloat, FP.Structural.toFloat,
    FP.Software.scalarFP32, FP.Software.fp32Context,
    Option.bind, ComputeDType.eraseDType]

/-- In particular, fneg turns positive zero into negative zero. -/
theorem negates_positive_zero :
    (FP.Value.zero .fp32).neg = FP.Value.zero .fp32 Bool.true := by decide +kernel

/-- Negative zero becomes positive zero, without arithmetic rounding. -/
theorem negates_negative_zero :
    (FP.Value.zero .fp32 Bool.true).neg = FP.Value.zero .fp32 := by decide +kernel

#axiomsClean single_runtime_negation
#axiomsClean negates_positive_zero
#axiomsClean negates_negative_zero

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.FloatNegativeScalar
