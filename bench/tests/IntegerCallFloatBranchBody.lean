import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerCallFloatBranchBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("Y", .fp32)]
  scalars := [("s", .fp32), ("x", .fp32), ("C", .i64)]
  constexpr := ["FLAG"]

def source (Y : RegionName) (C : Int) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [C := C] in triton {
    if FLAG:
        y = 0.0
    else:
        y = s * x / C
    tl.store(Y, y)
  }

/-- Both branches are floating. The runtime integer conversion remains an
integer-to-real operation rather than a floating literal substitution. -/
def expected (Y : RegionName) (C : Int) (FLAG : Bool) : ComputeKernel :=
  let body : List ComputeStmt := [
    .ifThenElse (.constBool FLAG)
      [.assign .real [] "y" (.compute (.full [] (.alg .fp32 (.const 0.0))))]
      [.assign .real [] "y" (.compute (.alg .fp32
        (.div .real .nil
          (.mul .real .nil (.ref .real [] "s") (.ref .real [] "x"))
          (.intToReal (.constInt C)))))],
    .store .real [] (.region Y (.constNat 0))
      (.compute (.alg .fp32 (.ref .real [] "y"))) .none]
  .mk [] [Y] body

theorem complete (Y : RegionName) (C : Int) (FLAG : Bool) :
    source Y C FLAG = expected Y C FLAG := rfl

#axiomsClean complete
end VeriTile.Tests.IntegerCallFloatBranchBody
