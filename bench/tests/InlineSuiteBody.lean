import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.InlineSuiteBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  constexpr := ["SCALE", "N"]

def inlineBody (X Y : RegionName) (SCALE : Bool) (N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    if SCALE: x = 2.0 * x
    else: x = x + 1.0
    for k in range(N): x = x + 1.0
    if x < 0.0: x = -x
    tl.store(Y, x)
  }

def indentedBody (X Y : RegionName) (SCALE : Bool) (N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    if SCALE:
        x = 2.0 * x
    else:
        x = x + 1.0
    for k in range(N):
        x = x + 1.0
    if x < 0.0:
        x = -x
    tl.store(Y, x)
  }

/-- Static branches, runtime branches and loop scope survive source spelling. -/
theorem statement_scope (X Y : RegionName) (SCALE : Bool) (N : Nat) :
    inlineBody X Y SCALE N = indentedBody X Y SCALE N := by
  cases SCALE <;> rfl

-- Slice and string colons cannot end a header. These are lexical checks,
-- independent of whether a later intrinsic compiler accepts the expression.
#guard DSL.PythonBody.render "{\n  if values[:, 0]: x = float(\"-inf\")\n}" ==
  .ok "if values[:, 0] {\nx = float(\"-inf\")\n}"
#guard DSL.PythonBody.render "{\n  if key == \":\": x = 1\n}" ==
  .ok "if key == \":\" {\nx = 1\n}"

-- An unsupported multi-statement suite must not silently move its second
-- statement outside the condition, including a semicolon after whitespace.
#guard match DSL.PythonBody.render "{\n  if flag: x = 1; y = 2\n}" with
  | .error _ => Bool.true
  | .ok _ => Bool.false

#axiomsClean statement_scope
end VeriTile.Tests.InlineSuiteBody
