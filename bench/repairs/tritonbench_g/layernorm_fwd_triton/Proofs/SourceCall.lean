import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SuccessfulCall

/-! Total source-order call evaluation. Every case is selected by the actual
result of an individual prefix or dispatch call. Failure carries its actual
post-call state; an entered context always runs leave. The proof-only route
reification is total after successful preparation and adds no runtime guard. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Wrapper
open Torch Address Memory Execution HostPrepared
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}
  {platform : Runtime.Platform α} {service : Runtime.Service platform api neg caller Host}
  {lib : APIHost api neg caller Host} {q : Request} {initial : State Store Host}

def closeContext (lib : APIHost api neg caller Host) (saved : Nat)
    (body : Outcome (State Store Host) Tensor) : Outcome (State Store Host) Tensor :=
  let last := (lib.leave saved).call.run body.state
  ⟨last.1,match last.2 with | .ok _ => body.result | .error code => .error (.api .exitDevice code),
    body.trace++[.exitDevice]⟩

def completedOutcome (setup : Setup lib q initial) : Outcome (State Store Host) Tensor :=
  let prefixTrace := ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).trace ++
    [.enterDevice] ++ ((HostPrelude.prepare lib q.x q.w).run setup.enterState).trace
  let body := (Runtime.dispatch platform api neg service setup.prepared q.eps).run setup.finish
  let out := closeContext lib initial.owner body
  {out with trace:=prefixTrace++out.trace}

/-- The constructors record only actual local results. The final outcome is
computed separately below, including failed context exit replacing an error. -/
inductive Evaluation (platform : Runtime.Platform α) (api : API α Store Byte) (neg : α → α)
    (service : Runtime.Service platform api neg caller Host) (lib : APIHost api neg caller Host)
    (q : Request) (initial : State Store Host) where
  | validationFailed (error : Failure)
      (failed : ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).result=.error error)
  | enterFailed (shape : Shape)
      (validated : ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).result=.ok shape)
      (code : Nat) (failed : ((lib.enter q.x.device).call.run initial).2=.error code)
  | preparationFailed (shape : Shape)
      (validated : ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).result=.ok shape)
      (entered : ((lib.enter q.x.device).call.run initial).2=.ok ())
      (error : Failure)
      (failed : ((HostPrelude.prepare lib q.x q.w).run ((lib.enter q.x.device).call.run initial).1).result=.error error)
  | dispatchFailed (setup : Setup lib q initial) (error : Failure)
      (failed : ((Runtime.dispatch platform api neg service setup.prepared q.eps).run setup.finish).result=.error error)
  | exitFailed (setup : Setup lib q initial)
      (route : Runtime.Route platform api neg service setup.prepared q.eps setup.finish)
      (code : Nat) (failed : ((lib.leave initial.owner).call.run (dispatchedState setup route)).2=.error code)
  | success (call : SuccessfulCall platform api neg service lib q initial)

/-- Exhaustive evaluation after the public contract checks succeed. The
retained shape validator splits the admitted rank/head/width facts; allocation,
compiler, launch and context-exit outcomes are the actual local call results. -/
def evaluate (domain : Domain q initial caller) : Evaluation platform api neg service lib q initial := by
  cases hv : ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).result with
  | error error => exact .validationFailed error hv
  | ok shape =>
    cases he : ((lib.enter q.x.device).call.run initial).2 with
    | error code => exact .enterFailed shape hv code he
    | ok done =>
      cases done
      cases hp : ((HostPrelude.prepare lib q.x q.w).run ((lib.enter q.x.device).call.run initial).1).result with
      | error error => exact .preparationFailed shape hv he error hp
      | ok values =>
        let route := Classical.choose (HostPrelude.prepare_route lib q.x q.w _ values hp)
        let setup : Setup lib q initial := ⟨domain,shape,hv,he,route⟩
        cases hd : ((Runtime.dispatch platform api neg service setup.prepared q.eps).run setup.finish).result with
        | error error => exact .dispatchFailed setup error hd
        | ok out =>
          let dispatch := Classical.choose (Runtime.dispatch_route out hd)
          cases hl : ((lib.leave initial.owner).call.run (dispatchedState setup dispatch)).2 with
          | error code => exact .exitFailed setup dispatch code hl
          | ok done => cases done;exact .success ⟨setup,dispatch,hl⟩

def Evaluation.outcome (ev : Evaluation platform api neg service lib q initial) : Outcome (State Store Host) Tensor :=
  match ev with
  | .validationFailed error _ =>
      let first := (HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial
      ⟨first.state,.error error,first.trace⟩
  | .enterFailed _ _ code _ =>
      ⟨((lib.enter q.x.device).call.run initial).1,.error (.api .enterDevice code),
        ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).trace++[.enterDevice]⟩
  | .preparationFailed _ _ _ error _ =>
      let pre := (HostPrelude.prepare lib q.x q.w).run ((lib.enter q.x.device).call.run initial).1
      let out := closeContext lib initial.owner ⟨pre.state,.error error,pre.trace⟩
      {out with trace:=((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run initial).trace++[.enterDevice]++out.trace}
  | .dispatchFailed setup _ _ => completedOutcome (service:=service) setup
  | .exitFailed setup _ _ _ => completedOutcome (service:=service) setup
  | .success c => completedOutcome (service:=service) c.setup

theorem completed_success (c : SuccessfulCall platform api neg service lib q initial) :
    (completedOutcome (service:=service) c.setup).result=.ok c.returned ∧
    (completedOutcome (service:=service) c.setup).state=c.final := by
  simp only [completedOutcome,closeContext,(dispatch_execution c.dispatch).1,(dispatch_execution c.dispatch).2]
  change (match ((lib.leave initial.owner).call.run c.kernelState).2 with
    | .ok _ => .ok c.returned | .error code => .error (Failure.api .exitDevice code))=
      (Except.ok c.returned : Except Failure Tensor) ∧ _
  have exited : ((lib.leave initial.owner).call.run c.kernelState).2=.ok () := c.exited
  rw [exited]
  exact ⟨rfl,rfl⟩

/-- Every successful evaluation yields the same successful-call derivation,
exact returned tensor and computed final state used by both public readers. -/
theorem evaluation_success (ev : Evaluation platform api neg service lib q initial) (out : Tensor)
    (ok : ev.outcome.result=.ok out) :
    ∃ c : SuccessfulCall platform api neg service lib q initial,
      out=c.returned ∧ ev.outcome.state=c.final := by
  cases ev with
  | validationFailed error failed => simp [Evaluation.outcome] at ok
  | enterFailed shape validated code failed => simp [Evaluation.outcome] at ok
  | preparationFailed shape validated entered error failed =>
    simp only [Evaluation.outcome,closeContext] at ok
    split at ok <;> contradiction
  | dispatchFailed setup error failed =>
    simp only [Evaluation.outcome,completedOutcome,closeContext,failed] at ok
    split at ok <;> contradiction
  | exitFailed setup route code failed =>
    simp only [Evaluation.outcome,completedOutcome,closeContext,(dispatch_execution route).2] at ok
    unfold dispatchedState at failed
    rw [failed] at ok
    contradiction
  | success c =>
    have hs := completed_success c
    exact ⟨c,Except.ok.inj (ok.symm.trans hs.1),hs.2⟩

theorem evaluation_frame (ev : Evaluation platform api neg service lib q initial)
    (domain : Domain q initial caller) (i : Nat) (hi : initial.backing i) :
    api.byte ev.outcome.state.memory i=api.byte initial.memory i := by
  have hc : caller i := by rw [domain.backing];exact hi
  have completeFrame (setup : Setup lib q initial) :
      api.byte (completedOutcome (service:=service) setup).state.memory i=api.byte initial.memory i :=
    ((lib.leave initial.owner).call.frame _ i hc).trans
      (((Runtime.dispatch platform api neg service setup.prepared q.eps).frame setup.finish i hc).trans
        (setup_frame setup i hc))
  cases ev with
  | validationFailed error failed => exact (HostPrelude.validate q.x q.w).frame initial i hc
  | enterFailed shape validated code failed => exact (lib.enter q.x.device).call.frame initial i hc
  | preparationFailed shape validated entered error failed =>
    exact ((lib.leave initial.owner).call.frame _ i hc).trans
      (((HostPrelude.prepare lib q.x q.w).frame _ i hc).trans ((lib.enter q.x.device).call.frame initial i hc))
  | dispatchFailed setup error failed => exact completeFrame setup
  | exitFailed setup route code failed => exact completeFrame setup
  | success c => exact completeFrame c.setup

/-- Assertion evaluation cannot modify host state before device entry. -/
theorem validation_state (s : State Store Host) :
    ((HostPrelude.validate (api:=api) (caller:=caller) q.x q.w).run s).state=s := by
  simp only [HostPrelude.validate,Action.bind,check,Action.pure]
  split <;> simp_all
  split <;> simp_all
  split <;> simp_all
  split <;> simp_all

/-- Any entered failing body is followed by exit on its completed-prefix
state; neither its allocation effects nor completed stores are rolled back. -/
theorem close_failure (body : Outcome (State Store Host) Tensor) (error : Failure)
    (failed : body.result=.error error) :
    (closeContext lib initial.owner body).state=((lib.leave initial.owner).call.run body.state).1 ∧
    (closeContext lib initial.owner body).trace=body.trace++[.exitDevice] ∧
    (closeContext lib initial.owner body).result=
      match ((lib.leave initial.owner).call.run body.state).2 with
      | .ok _ => .error error | .error code => .error (.api .exitDevice code) := by
  simp [closeContext,failed]

def sourceCall (domain : Domain q initial caller) : Outcome (State Store Host) Tensor :=
  (evaluate (platform:=platform) (service:=service) (lib:=lib) domain).outcome

theorem source_success (domain : Domain q initial caller) (out : Tensor)
    (ok : (sourceCall (service:=service) (lib:=lib) domain).result=.ok out) :
    ∃ c : SuccessfulCall platform api neg service lib q initial,
      out=c.returned ∧ (sourceCall (service:=service) (lib:=lib) domain).state=c.final :=
  evaluation_success _ out ok

theorem source_frame (domain : Domain q initial caller) (i : Nat) (hi : initial.backing i) :
    api.byte (sourceCall (service:=service) (lib:=lib) domain).state.memory i=api.byte initial.memory i :=
  evaluation_frame _ domain i hi

#axiomsClean evaluate
#axiomsClean completed_success
#axiomsClean evaluation_success
#axiomsClean evaluation_frame
#axiomsClean validation_state
#axiomsClean close_failure
#axiomsClean source_success
#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Wrapper
end VeriTile.Bench.Repairs.LayernormFwdTriton
