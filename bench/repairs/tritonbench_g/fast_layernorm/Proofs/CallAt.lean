import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReceiptAction

/-! A call at its actual initial state. The continuation receives a proof of
its preceding successful result; this is erased evidence, not a new check. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.CallAt
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
variable {S Key Value α β : Type} {world : World S Key Value}

structure Call (world : World S Key Value) (initial : S) (α : Type) where
  outcome : Outcome S α
  frame : ∀ key,world.caller key → world.observe outcome.state key=world.observe initial key

def ofAction (action : Action world α) (initial : S) : Call world initial α :=
  ⟨action.run initial,action.frame initial⟩

def followOutcome (state : S) (trace : List Stage) (result : Except Failure α)
    (f : (out : α) → result=.ok out → Outcome S β) : Outcome S β := by
  cases result with
  | error e => exact ⟨state,.error e,trace⟩
  | ok out => exact ⟨(f out rfl).state,(f out rfl).result,trace++(f out rfl).trace⟩

theorem follow_frame (state : S) (trace : List Stage) (result : Except Failure α)
    (f : (out : α) → result=.ok out → Outcome S β) (key : Key)
    (frame : ∀ out h,world.observe (f out h).state key=world.observe state key) :
    world.observe (followOutcome state trace result f).state key=world.observe state key := by
  cases result with
  | error e => rfl
  | ok out => exact frame out rfl

theorem follow_success (state : S) (trace : List Stage) (result : Except Failure α)
    (f : (out : α) → result=.ok out → Outcome S β) (out : β)
    (ok : (followOutcome state trace result f).result=.ok out) :
    ∃ first h,(f first h).result=.ok out ∧ (followOutcome state trace result f).state=(f first h).state := by
  cases result with
  | error e => cases ok
  | ok first => exact ⟨first,rfl,ok,rfl⟩

def bind {initial : S} (a : Call world initial α)
    (f : (out : α) → a.outcome.result=.ok out → Call world a.outcome.state β) : Call world initial β where
  outcome := followOutcome a.outcome.state a.outcome.trace a.outcome.result (fun out h => (f out h).outcome)
  frame key hk := (follow_frame _ _ _ _ key (fun out h => (f out h).frame key hk)).trans (a.frame key hk)

theorem bind_success {initial : S} (a : Call world initial α)
    (f : (out : α) → a.outcome.result=.ok out → Call world a.outcome.state β)
    (out : β) (ok : (bind a f).outcome.result=.ok out) :
    ∃ first h,(f first h).outcome.result=.ok out ∧ (bind a f).outcome.state=(f first h).outcome.state :=
  follow_success _ _ _ _ out ok

#axiomsClean bind_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.CallAt
