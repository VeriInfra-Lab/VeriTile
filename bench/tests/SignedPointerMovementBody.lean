import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedPointerMovementBody
open VeriTile.Triton FP.Structural
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.fp32)]
  scalars := [("delta",.i64)]
  constexpr := ["BASE"]

def source (X : RegionName) (BASE : Nat) (delta : Int) : ComputeKernel :=
  with_triton_call signature call_args [delta := delta] in triton {
    p = X + BASE
    p += delta
    left = p + delta
    right = delta + p
    backward = p - delta
  }

private theorem projected_sum (base : Nat) (delta : Int) :
    base + delta.toNat - (-delta).toNat = ((base : Int) + delta).toNat := by
  omega

/-- Actual interpreter results cover both signs, zero, and projection of an
out-of-region result. Valid dereferences still require the caller's bounds. -/
theorem movements {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (X : RegionName) (base : Nat) (delta : Int) :
    ∃ t, exec M (source X base delta) s = some t ∧
      t.regs .ptr [] "p" = some (fun _ => (X,((base : Int)+delta).toNat)) ∧
      t.regs .ptr [] "left" = some (fun _ => (X,((((base : Int)+delta).toNat : Int)+delta).toNat)) ∧
      t.regs .ptr [] "right" = some (fun _ => (X,((((base : Int)+delta).toNat : Int)+delta).toNat)) ∧
      t.regs .ptr [] "backward" = some (fun _ => (X,((((base : Int)+delta).toNat : Int)-delta).toNat)) := by
  simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,State.setReg,Region.cast,numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,projected_sum,Int.sub_eq_add_neg]
  all_goals repeat' apply And.intro
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,projected_sum]
  all_goals omega

def conditionalSignature : CallSignature where
  pointers := [("X",.fp32)]
  constexpr := ["BASE","K","REVERSE"]

def conditional (X : RegionName) (BASE : Nat) (K : Int) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call conditionalSignature call_args [] in triton {
    p = X + BASE
    p += -K if REVERSE else K
  }

theorem conditional_address {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (X : RegionName) (base : Nat) (stride : Int) (reverse : Bool) :
    (exec M (conditional X base stride reverse) s).bind (fun t => t.regs .ptr [] "p") =
      some (fun _ => (X,((base : Int)+(if reverse then -stride else stride)).toNat)) := by
  cases reverse <;>
    simp [conditional,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,State.setReg,Region.cast,numeric,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,projected_sum]
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,projected_sum]
  all_goals omega

theorem reverse_address {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    (exec M (conditional "X" 12 4 Bool.true) s).bind (fun t => t.regs .ptr [] "p") =
      some (fun _ => (("X" : RegionName),8)) := by
  simpa using conditional_address M s "X" 12 4 Bool.true

def subtract (X : RegionName) (BASE : Nat) (delta : Int) : ComputeKernel :=
  with_triton_call signature call_args [delta := delta] in triton {
    p = X + BASE
    p -= delta
  }

/-- Subtraction must keep the sign of the runtime operand as well. -/
theorem subtract_address {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (X : RegionName) (base : Nat) (delta : Int) :
    (exec M (subtract X base delta) s).bind (fun t => t.regs .ptr [] "p") =
      some (fun _ => (X,((base : Int)-delta).toNat)) := by
  simp [subtract,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,State.setReg,Region.cast,numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,projected_sum,Int.sub_eq_add_neg]
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,projected_sum]
  all_goals omega

#axiomsClean subtract_address

#axiomsClean movements
#axiomsClean conditional_address
#axiomsClean reverse_address
end VeriTile.Tests.SignedPointerMovementBody
