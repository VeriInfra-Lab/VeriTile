import tests.lean.SingleAssignmentPointerOffsetReference
import VeriTile.Triton.DSL.SingleAssignmentPointerOffsets
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SingleAssignmentPointerOffsets
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("delta",.i64)]
  constexpr := []
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (base : Nat) (delta : Int) : ComputeKernel :=
  with_triton_call signature call_args [delta := delta] pointer_offsets [X := base] in triton {
    X = X + delta
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- Negative displacements remain signed across the caller base addition. -/
theorem complete_program : source=SingleAssignmentPointerOffsetReference.source := by rfl

theorem executes {A : Type} [Inhabited A] (M : Algebra A) (X Y : RegionName)
    (base : Nat) (delta : Int) (s : State A) :
    ∃ t, exec M (source X Y base delta) s=some t ∧
      t.mem Y 0=.mk .real (M.fp32Load ((s.mem X ((base : Int)+delta).toNat).read .real)) := by
  simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,Option.bind,numeric,bop,Region.cast,
    ComputeDType.eraseDType,State.setReg,State.write]

def runtime (X Y : RegionName) (base : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [X := base] in triton {
    X = X + delta
    value = tl.load(X)
    tl.store(Y, value)
  }

theorem runtime_complete_program : runtime=SingleAssignmentPointerOffsetReference.runtime := by rfl

/-- Signed runtime inputs, including displacements below zero, are retained. -/
theorem runtime_executes {A : Type} [Inhabited A] (M : Algebra A) (X Y : RegionName)
    (base : Nat) (delta : Int) (s : State A) (input : s.regs .int [] "delta"=some (fun _=>delta)) :
    ∃ t, exec M (runtime X Y base) s=some t ∧
      t.mem Y 0=.mk .real (M.fp32Load ((s.mem X ((base : Int)+delta).toNat).read .real)) := by
  simp [runtime,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,Option.bind,numeric,bop,Region.cast,
    ComputeDType.eraseDType,State.setReg,State.write,input]

/-- error: an offset pointer may be assigned only once -/
#guard_msgs (error) in
example (X Y : RegionName) (base : Nat) (delta : Int) : ComputeKernel :=
  with_triton_call signature call_args [delta := delta] pointer_offsets [X := base] in triton {
    X = X + delta
    X += 1
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- error: an offset pointer requires one top-level self-add binding -/
#guard_msgs (error) in
example (X Y : RegionName) (base : Nat) (delta : Int) (flag : Bool) : ComputeKernel :=
  with_triton_call signature call_args [delta := delta] pointer_offsets [X := base] in triton {
    if flag:
      X = X + delta
    value = tl.load(X)
    tl.store(Y, value)
  }

#axiomsClean complete_program
#axiomsClean executes
#axiomsClean runtime_complete_program
#axiomsClean runtime_executes
end
end VeriTile.Tests.SingleAssignmentPointerOffsets
