import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonConditionalAssignmentBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  scalars := [("a", .i64), ("b", .i64), ("c", .i64), ("d", .i64)]
  constexpr := ["FLAG"]

def source (X Y : RegionName) (a b c d : Int) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [a := a, b := b, c := c, d := d] in triton {
    x = tl.load(X)
    if a < b and b <= c and c != d:
        y = x if FLAG else 0.0
        if x < 0.0:
            y = -y
    else:
        y = -x
    tl.store(Y, y)
  }

def parenthesized (X Y : RegionName) (a b c d : Int) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [a := a, b := b, c := c, d := d] in triton {
    x = tl.load(X)
    if ((a < b) and (b <= c)) and (c != d):
        y = (x if FLAG else 0.0)
        if x < 0.0:
            y = -y
    else:
        y = -x
    tl.store(Y, y)
  }

/-- Comparisons bind before the left-associated conjunction. The assignment
conditional ends on its Python logical line, before the next runtime branch. -/
theorem whole_body (X Y : RegionName) (a b c d : Int) (FLAG : Bool) :
    source X Y a b c d FLAG = parenthesized X Y a b c d FLAG := rfl

theorem statements (X Y : RegionName) (a b c d : Int) (FLAG : Bool) :
    (source X Y a b c d FLAG).surfaceBody.length = 3 := rfl

#guard DSL.PythonBody.render "{\n  y = x if flag else 0.0\n  if other: y = -y\n}" ==
  .ok "y = (x if flag else 0.0)\nif other {\ny = -y\n}"
#guard DSL.PythonBody.render "{\n  y = f(\" if \", x == z)\n}" ==
  .ok "y = f(\" if \", x == z)"
#guard DSL.PythonBody.render "{\n  y = (x if flag else 0.0)\n}" ==
  .ok "y = (x if flag else 0.0)"
#guard DSL.PythonBody.render "{\n  y = x if a == b else z\n}" ==
  .ok "y = (x if a == b else z)"

#axiomsClean whole_body
#axiomsClean statements
end VeriTile.Tests.PythonConditionalAssignmentBody
