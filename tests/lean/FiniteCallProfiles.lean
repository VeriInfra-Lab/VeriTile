import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FiniteCallProfiles
open VeriTile.Triton FP.Structural

inductive Keep where | integer | boolean
inductive Output where | small | wide

def signature (keeping : Keep) (storage : Output) : CallSignature where
  pointers := [("keep",match keeping with | .integer => .i32 | .boolean => .boolean),
    ("output",match storage with | .small => .i8 | .wide => .i32)]

def source (keeping : Keep) (storage : Output) (keep output : RegionName) : ComputeKernel :=
  with_triton_signature (signature keeping storage) in triton {
    enabled = tl.load(keep)
    value = tl.where(enabled, tl.cast(7, tl.int32), tl.cast(13, tl.int32))
    tl.store(output, value)
  }

/-- Caller dtype choices select the actual typed load and destination cast.
Both mask representations and both integer storage widths execute the same
source body, including when the two ports alias. -/
theorem selected_execution {α : Type} [Inhabited α] (M : Algebra α)
    (keeping : Keep) (storage : Output) (keep output : RegionName)
    (enabled : Bool) (s : State α)
    (input : s.mem keep 0 = match keeping with
      | .integer => Cell.mk .int (if enabled then 1 else 0)
      | .boolean => Cell.mk .bool enabled) :
    ∃ final, exec M (source keeping storage keep output) s = some final ∧
      final.mem output 0 = Cell.mk .int (if enabled then 7 else 13) := by
  cases keeping <;> cases storage <;> cases enabled
  all_goals
    simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,bop,numeric,numericEq,numericLt,natLt,store,Region.cast,
      State.setReg,State.write,TileShape.allIndices,input,integralMod,IntegerExpr.sameShape,
      IntegerExpr.same_left,IntegerExpr.same_right,IntegerExpr.signedBits]

/-- A runtime size is not a finite caller dtype profile. It must not be
silently specialized into one of its possible storage types. -/
def dynamicSignature (n : Nat) : CallSignature where
  pointers := [("output",if n = 0 then .i32 else .boolean)]

/-- error: call signature dtypes must be statically known -/
#guard_msgs in
example (n : Nat) (output : RegionName) : ComputeKernel :=
  with_triton_signature (dynamicSignature n) in triton {
    tl.store(output, 1)
  }

#axiomsClean selected_execution
end VeriTile.Tests.FiniteCallProfiles
