import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ReshapePrecision
open VeriTile.Triton

/-- Independently written typed programs: the coordinate-only view and the
following addition retain the caller's computation precision. -/
def reference (wide : Bool) : ComputeKernel :=
  if wide then .mk [] [] [
    .assign .real [1] "flat" (.compute (.alg .fp64 (.reshape [1] (.ref .real [] "value")))),
    .assign .real [1] "result" (.compute (.alg .fp64 (.add .real (.consSame .nil)
      (.ref .real [1] "flat") (.ref .real [1] "flat"))))]
  else .mk [] [] [
    .assign .real [1] "flat" (.compute (.alg .fp32 (.reshape [1] (.ref .real [] "value")))),
    .assign .real [1] "result" (.compute (.alg .fp32 (.add .real (.consSame .nil)
      (.ref .real [1] "flat") (.ref .real [1] "flat"))))]

def fp32 : ComputeKernel := triton {
  tl.param value : tl.float32
  flat = tl.reshape(value, [1])
  result = flat + flat
}
def fp64 : ComputeKernel := triton {
  tl.param value : tl.float64
  flat = tl.reshape(value, [1])
  result = flat + flat
}

theorem fp32_program : fp32 = reference Bool.false := rfl
theorem fp64_program : fp64 = reference Bool.true := rfl

/-- An inline cast keeps the source's FP32 context when the view's result
has FP64 precision. The reshape must not erase this nested cast context. -/
def castSource : ComputeKernel := triton {
  flat = tl.reshape((1.0).to(tl.float32).to(tl.float64), [1])
}
def castReference : ComputeKernel := .mk [] [] [
  .assign .real [1] "flat" (.compute (.alg .fp64 (.reshape [1]
    (.castFloat .real .real (.withPrecision .real .fp32 (.castFloat .real .real (.const (1.0))))))))]
theorem cast_program : castSource = castReference := rfl

#axiomsClean fp32_program
#axiomsClean fp64_program
#axiomsClean cast_program
#auditModuleAxioms
end VeriTile.Tests.ReshapePrecision
