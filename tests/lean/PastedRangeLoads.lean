import VeriTile.Triton.DSL.PastedRangeLoads
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Tests.PastedRangeLoads
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- One-argument ranges preserve failures, every reached index and all state
for an arbitrary arithmetic backend and arbitrary loop body. -/
theorem loop_eq_unit_range {α : Type} [Inhabited α] (M : Algebra α)
    (index : RegName) (cur stop : Nat) (body : List ComputeStmt) (s : State α) :
    loop M index cur stop body s=range M index cur stop 1 body s := by
  generalize hm : stop-cur=remaining
  induction remaining using Nat.strong_induction_on generalizing cur s with
  | h remaining ih=>
    by_cases active : cur<stop
    · rw [loop,range,if_neg (by decide : ¬(1 : Nat)=0),if_pos active,if_pos active]
      cases ran : run M body (s.setReg index .nat [] (fun _=>cur)) with
      | none=>rfl
      | some t=>
        change loop M index (cur+1) stop body t=range M index (cur+1) stop 1 body t
        exact ih (stop-(cur+1)) (by omega) (cur+1) t rfl
    · rw [loop,range,if_neg (by decide : ¬(1 : Nat)=0),if_neg active,if_neg active]

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  readPorts := ["X"]
  writePorts := ["Y"]

def augmented (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    acc = tl.full((), 0.0, tl.float32)
    acc += tl.load(X)
    tl.store(Y, acc)
  }

def ordinary (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    acc = tl.full((), 0.0, tl.float32)
    acc = acc + tl.load(X)
    tl.store(Y, acc)
  }

/-- Both source forms retain the same typed load before the add. -/
theorem augmented_is_typed (X Y : RegionName) : augmented X Y=ordinary X Y := rfl

def typedLoads (body : List ComputeStmt) : Nat :=
  body.foldl (fun count statement=>match statement with
    | .assign _ _ _ (.compute (.load .fp32 _ _))=>count+1
    | _=>count) 0

theorem one_typed_load (X Y : RegionName) : typedLoads (augmented X Y).surfaceBody=1 := rfl

#axiomsClean loop_eq_unit_range
#axiomsClean augmented_is_typed
#axiomsClean one_typed_load
end
end VeriTile.Tests.PastedRangeLoads
