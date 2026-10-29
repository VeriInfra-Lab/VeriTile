import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedLoadCast
open VeriTile.Triton

def direct (labels : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region labels = tl.int32, x = tl.float32, y = tl.float32
  labels += tl.program_id(0)
  label = tl.load(labels).to(tl.int32)
  if label != $((-100 : Int)) {
    v = tl.load(x + label)
    tl.store(y, v)
  }
}
def parenthesized (labels : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region labels = tl.int32, x = tl.float32, y = tl.float32
  labels += tl.program_id(0)
  label = (tl.load(labels)).to(tl.int32)
  if label != $((-100 : Int)) {
    v = tl.load(x + label)
    tl.store(y, v)
  }
}
/-- Direct-call syntax retains the same signed label and live negative
sentinel as the parenthesized spelling, even when used as a gather address. -/
example (labels : Region .int) (x y : RegionName) : direct labels x y = parenthesized labels x y := rfl

private def hasSignedSentinelGuard : ComputeStmt → Bool
  | .ifThen (.ne .int _ (.ref .int [] "label") (.constInt (-100))) _ => Bool.true
  | _ => Bool.false

/-- Address inference must not reinterpret the live negative sentinel as Nat. -/
example (labels : Region .int) (x y : RegionName) :
    (direct labels x y).surfaceBody.getLast?.map hasSignedSentinelGuard =
      some Bool.true := rfl

/-- Copies and mixed-width selection remain signed when later used as a
gather address or an in-place pointer offset. -/
def copied (labels : Region .int) (x y : RegionName) : ComputeKernel := triton {
  tl.region labels = tl.int32, x = tl.float32, y = tl.float32
  raw = tl.load(labels)
  label = raw
  selected = tl.where(label != $((-100 : Int)), label, $((-100 : Int)))
  if selected != $((-100 : Int)) {
    x += selected
    v = tl.load(x)
    tl.store(y, v)
  }
}

private def assignsSignedLabel : ComputeStmt → Bool
  | .assign .int [] "label" (.alg (.ref .int [] "raw")) => Bool.true
  | _ => Bool.false

example (labels : Region .int) (x y : RegionName) :
    (copied labels x y).surfaceBody[1]?.map assignsSignedLabel = some Bool.true := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.SignedLoadCast
