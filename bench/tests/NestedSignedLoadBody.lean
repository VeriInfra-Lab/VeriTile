import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.NestedSignedLoadBody
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def signature32 : CallSignature where
  pointers := [("P",.i32),("X",.fp32)]
def signature64 : CallSignature where
  pointers := [("P",.i64),("X",.fp32)]

def source32 (P X : RegionName) : ComputeKernel :=
  with_triton_signature signature32 in triton {
    offset = tl.arange(0, 2) + tl.load(P)
    X = X + offset
  }
def source64 (P X : RegionName) : ComputeKernel :=
  with_triton_signature signature64 in triton {
    offset = tl.arange(0, 2) + tl.load(P)
    X = X + offset
  }

/-- Both source widths use the declared signed memory channel in this
mathematical index representation; byte decoding/range obligations are caller IO. -/
theorem signed_channels (P X : RegionName) : source32 P X = source64 P X := rfl

/-- Address use does not change the loaded channel or truncate a negative
intermediate. Only the final pointer conversion projects to natural offsets. -/
theorem signed_offset {α : Type} [Inhabited α] (M : Algebra α)
    (P X : RegionName) (s : State α) :
    ∃ t,exec M (source64 P X) s=some t ∧
      t.regs .int [2] "offset"=some (fun i => (i.1.val : Int)+Cell.read .int (s.mem P 0)) := by
  simp [source64,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,State.setReg,Option.bind,numeric,bop,Region.cast]
  funext i
  rfl

#axiomsClean signed_channels
#axiomsClean signed_offset
end VeriTile.Tests.NestedSignedLoadBody
