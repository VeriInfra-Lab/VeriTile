import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Host

/-! Receipt attachment does not insert a guard or a possible failure. It adds
only facts proved from a successful action's actual result. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
variable {S Key Value α : Type} {world : World S Key Value}

def certifiedResult (r : Except Failure α) (P : α → Prop)
    (proof : ∀ out,r=.ok out → P out) : Except Failure {out // P out} := by
  cases r with
  | error e => exact .error e
  | ok out => exact .ok ⟨out,proof out rfl⟩

theorem certifiedResult_forget (r : Except Failure α) (P : α → Prop)
    (proof : ∀ out,r=.ok out → P out) :
    (certifiedResult r P proof).map Subtype.val=r := by
  cases r <;> rfl

def certify (a : Action world α) (P : α → Prop)
    (proof : ∀ s out,(a.run s).result=.ok out → P out) : Action world {out // P out} where
  run s := ⟨(a.run s).state,certifiedResult (a.run s).result P (proof s),(a.run s).trace⟩
  frame := a.frame

theorem certify_state (a : Action world α) (P : α → Prop)
    (proof : ∀ s out,(a.run s).result=.ok out → P out) (s : S) :
    ((certify a P proof).run s).state=(a.run s).state := rfl

theorem certify_success (a : Action world α) (P : α → Prop)
    (proof : ∀ s out,(a.run s).result=.ok out → P out) (s : S) (out : {out // P out})
    (ok : ((certify a P proof).run s).result=.ok out) : (a.run s).result=.ok out.val := by
  have h := congrArg (Except.map Subtype.val) ok
  exact (certifiedResult_forget _ _ _).symm.trans h

def getState : Action world S where
  run s := ⟨s,.ok s,[]⟩
  frame _ _ _ := rfl

/-- Extraction for the reused device-context action, including the actual exit
state. The ordinary leave call is run on body failure as well. -/
theorem device_success (enter leave : Local world Unit) (body : Action world α)
    (s : S) (out : α) (ok : ((inDevice enter leave body).run s).result=.ok out) :
    (enter.run s).2=.ok () ∧
    (body.run (enter.run s).1).result=.ok out ∧
    (leave.run (body.run (enter.run s).1).state).2=.ok () ∧
    ((inDevice enter leave body).run s).state=(leave.run (body.run (enter.run s).1).state).1 := by
  cases he : (enter.run s).2 with
  | error e => simp [inDevice,Action.bind,call,he,Except.mapError] at ok
  | ok unit =>
    cases unit
    cases hl : (leave.run (body.run (enter.run s).1).state).2 with
    | error e => simp [inDevice,Action.bind,call,he,hl,Except.mapError] at ok
    | ok unit =>
      cases unit
      refine ⟨rfl,?_,rfl,?_⟩
      · simpa [inDevice,Action.bind,call,he,hl,Except.mapError] using ok
      · simp [inDevice,Action.bind,call,he,hl,Except.mapError]

#axiomsClean certify_success
#axiomsClean certify_state
#axiomsClean device_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ReceiptAction
