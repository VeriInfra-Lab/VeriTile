import VeriTile.Triton.DSL.ImplicitAtomicAdd
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ImplicitAtomicAdd
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp32)]
  readPorts := ["X","Y"]
  writePorts := ["Y"]

def implicit (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    value = tl.load(X)
    target = Y + 0
    tl.atomic_add(target, value)
  }

def explicit (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    value = tl.load(X)
    target = Y + 0
    tl.atomic_add(target, value.to(tl.float32))
  }

theorem explicit_cast_ast : implicit "X" "Y" = explicit "X" "Y" := rfl

/-- Only the FP32 atomic primitive is supplied. The actual FP16 input is
widened before adding to the old FP32 cell, under an arbitrary backend. -/
theorem execution {α : Type} [Inhabited α] (M : Algebra α) (add : α→α→α)
    (supported : M.atomicAdd .real=some add) (s : State α) :
    ∃ t, exec M (implicit "X" "Y") s=some t ∧
      t.mem "Y" 0=.mk .real (add ((s.mem "Y" 0).read .real)
        (M.cast (some .fp32) .fp16 .real ((s.mem "X" 0).read .fp16))) := by
  simp [implicit,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,atomicAdd,atomicAddition,numeric,bop,Option.bind,
    supported,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.write,
    Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean explicit_cast_ast
#axiomsClean execution
end
end VeriTile.Tests.ImplicitAtomicAdd
