import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! External scalar declarations retain the precision of uses, emit no
assignment/cast/literal, and reject accidental redeclarations. Invocation
arguments, rather than a fabricated source assignment, supply their values. -/
namespace RuntimeScalarArgumentDSLTests
open VeriTile.Triton

def external32 := triton {
  tl.param eps : tl.float32
  y = eps + 1.0
}

theorem fp32_reads_argument : external32.surfaceBody = [
    .assign .real [] "y" (.compute (.alg .fp32
      (.add .real .nil (.ref .real [] "eps") (.const 1.0))))] := rfl

def external64 := triton {
  tl.param eps : tl.float64
  y = eps + 1.0
}

theorem fp64_reads_argument : external64.surfaceBody = [
    .assign .real [] "y" (.compute (.alg .fp64
      (.add .real .nil (.ref .real [] "eps") (.const 1.0))))] := rfl

def declarationOnly := triton {
  tl.param eps : tl.float32
}

example : declarationOnly.surfaceBody = [] := rfl

def externalInt64 := triton {
  tl.param total_classes : tl.int64
  classes = total_classes
}

/-- An incoming signed index is read directly, without a dummy assignment
or a conversion through floating point. -/
theorem int64_reads_argument : externalInt64.surfaceBody = [
    .assign .int [] "classes" (.alg (.ref .int [] "total_classes"))] := rfl

example : (triton { tl.param total_classes : tl.int64 }).surfaceBody = [] := rfl

def typedFP32Argument := triton {
  tl.param divfactor : tl.float32 (carrier=tl.float32)
  scaled = 2.0 * divfactor
}

/-- Read the actual FP32 register and retain its explicit algorithm-view
conversion under FP32 computation, as required by the dequantization ABI. -/
theorem fp32_word_reads_argument : typedFP32Argument.surfaceBody = [
    .assign .real [] "scaled" (.compute (.alg .fp32
      (.mul .real .nil (.const 2.0)
        (.castFloat .fp32 .real (.ref .fp32 [] "divfactor")))))] := rfl

example : (triton {
  tl.param divfactor : tl.float32 (carrier=tl.float32)
}).surfaceBody = [] := rfl

def mixedFloatInteger := triton {
  tl.param gain : tl.float32
  tl.param n : tl.int64
  output = gain / n
}

/-- An integer ABI argument must not force a floating computation into the
integer channel. This is the cross-entropy runtime class-count pattern. -/
theorem mixed_argument_precision : mixedFloatInteger.surfaceBody = [
    .assign .real [] "output" (.compute (.alg .fp32
      (.div .real .nil (.ref .real [] "gain")
        (.intToReal (.ref .int [] "n")))))] := rfl

/-- error: tl.param: register already bound: eps -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.param eps : tl.float32
  tl.param eps : tl.float64
}

/-- error: tl.param: register already bound: eps -/
#guard_msgs in
example : ComputeKernel := triton {
  eps = 1.0
  tl.param eps : tl.float32
}

/-- error: tl.param: only FP32, FP64 and Int64 scalar arguments are supported -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.param eps : tl.int32
}

/-- error: tl.param: explicit carrier requires FP32 precision and FP32 carrier -/
#guard_msgs in
example : ComputeKernel := triton {
  tl.param eps : tl.float64 (carrier=tl.float32)
}

#guard_msgs (drop info) in
#auditModuleAxioms
end RuntimeScalarArgumentDSLTests
