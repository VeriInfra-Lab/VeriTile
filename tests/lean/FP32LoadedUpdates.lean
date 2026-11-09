import VeriTile.Triton.DSL.PythonBlockAdvances
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.FP32LoadedUpdates
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

def abi : CallSignature where
  floatingPointers := [("X",.fp32)]
  readPorts := ["X"]
def source (X : RegionName) : ComputeKernel :=
  with_python_block_advances in
  with_triton_call abi call_args [] in triton {
    ptr = tl.make_block_ptr(base=X, shape=(1, 1), strides=(1, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    left = tl.load(ptr)
    right = tl.load(ptr)
    value = tl.load(ptr)
    value += tl.dot(left, right)
  }
def input {α : Type} [Inhabited α] (s : State α) : Values α .fp32 [1,1] :=
  fun _ => (s.mem "X" 0).read .fp32

/-- A scalar-result primitive stub is supplied at its actual typed FP32 port.
Both storage-to-compute conversions remain explicit for arbitrary algebras. -/
theorem update_ports {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (dot : α)
    (available : ∀ left right : Values α .fp32 [1,1],
      M.dotTyped (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[]) (m:=1) (k:=1) (n:=1)
        left right=some (fun _=>dot)) :
    ∃ t, exec M (source "X") s=some t ∧ t.mem=s.mem ∧
      t.regs .real [1,1] "value"=some (fun _=>
        M.binary (some .fp32) .real .add
          (M.cast (some .fp32) .fp32 .real ((s.mem "X" 0).read .fp32))
          (M.cast (some .fp32) .fp32 .real dot)) := by
  simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,ComputeDType.eraseDType,Option.bind,numeric,bop,
    State.setReg,Region.cast,ofFloat,toFloat,input,available]
  rfl

#axiomsClean update_ports
end
end VeriTile.Tests.FP32LoadedUpdates
