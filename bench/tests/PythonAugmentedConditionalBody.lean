import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonAugmentedConditionalBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  constexpr := ["REVERSE", "K"]

def source (X Y : RegionName) (REVERSE : Bool) (K : Int) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    p += -K if REVERSE else K
    x = tl.load(p)
    x += 1.0 if REVERSE else 2.0
    x -= 2.0 if REVERSE else 3.0
    x *= 3.0 if REVERSE else 4.0
    x /= 4.0 if REVERSE else 5.0
    tl.store(Y, x)
  }

def parenthesized (X Y : RegionName) (REVERSE : Bool) (K : Int) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    p += (-K if REVERSE else K)
    x = tl.load(p)
    x += (1.0 if REVERSE else 2.0)
    x -= (2.0 if REVERSE else 3.0)
    x *= (3.0 if REVERSE else 4.0)
    x /= (4.0 if REVERSE else 5.0)
    tl.store(Y, x)
  }

theorem whole_body (X Y : RegionName) (REVERSE : Bool) (K : Int) :
    source X Y REVERSE K = parenthesized X Y REVERSE K := rfl

#guard DSL.PythonBody.render "{\n  p += -K if flag else K\n  if other: p -= K\n}" ==
  .ok "p += (-K if flag else K)\nif other {\np -= K\n}"
#guard DSL.PythonBody.render "{\n  x*= a if b == c else d\n}" ==
  .ok "x*= (a if b == c else d)"
#guard DSL.PythonBody.render "{\n  x /= a if b != c else d\n}" ==
  .ok "x /= (a if b != c else d)"
#guard DSL.PythonBody.render "{\n  x -= f(\" if \", a == b)\n}" ==
  .ok "x -= f(\" if \", a == b)"
#guard DSL.PythonBody.render "{\n  x += (a if flag else b)\n}" ==
  .ok "x += (a if flag else b)"
#guard DSL.PythonBody.render "{\n  x = a if flag else b\n}" ==
  .ok "x = (a if flag else b)"
#guard DSL.PythonBody.render "{\n  if x == y: x += 1\n}" ==
  .ok "if x == y {\nx += 1\n}"

#axiomsClean whole_body
end VeriTile.Tests.PythonAugmentedConditionalBody
