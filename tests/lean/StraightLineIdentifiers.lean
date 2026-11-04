import VeriTile.Triton.DSL.StraightLineIdentifiers
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Tests.StraightLineIdentifiers
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    do = tl.load(X).to(
        tl.float32
    )
    tl.store(Y, do)
  }

/-- The reserved Python variable name and multiline cast compile to the full
source operations. Load materialization and the explicit source
conversion remain for an arbitrary arithmetic backend. -/
theorem copied_executes {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (s : State α) :
    ∃ t, exec (engine M) (copied X Y) s=some t ∧
      t.mem Y 0=.mk .real (M.cast (some .fp32) .real .real
        (M.fp32Load ((s.mem X 0).read .real))) := by
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [FP.Structural.exec,copied,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,evalOffsets,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,FP.Structural.store,State.setReg]
    rfl
  · simp [State.write,ofFloat,toFloat]

#axiomsClean copied_executes
end
end VeriTile.Tests.StraightLineIdentifiers
