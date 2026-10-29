import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardReal
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.CallAt

/-! Backward begins with the actual saved-tensor accessor. Its context identity
and structural metadata come from the preceding forward save. No gradient or
saved-statistics value equation is assumed here. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}

theorem context_identifies (lib : APIHost api neg inputs Metadata) (identity : Nat)
    (c : SavedContext) (initial : State Store Metadata) (lookup : lib.context initial.metadata identity=some c)
    (actual : SavedContext) (ok : ((call .saveOriginal (lib.accessSaved identity).call).run initial).result=.ok actual) : actual=c := by
  have h := lib.accessed identity initial actual ((call_ok _ _ _ _).mp ok)
  exact Option.some.inj (h.symm.trans lookup)

def invoke (lib : APIHost api neg inputs Metadata)
    (service : BackwardRuntime.Service platform api neg inputs Metadata) (identity : Nat)
    (c : SavedContext) (saved : SavedLayout c) (dy : Storage.Tensor) (initial : State Store Metadata)
    (lookup : lib.context initial.metadata identity=some c) :
    CallAt.Call (Host.world api inputs) initial GradientReturn.Gradients :=
  CallAt.bind (CallAt.ofAction (call .saveOriginal (lib.accessSaved identity).call) initial) fun actual accessed =>
    let same := context_identifies lib identity c initial lookup actual accessed
    let structural : SavedLayout actual := by simpa only [same] using saved
    CallAt.ofAction (inDevice (lib.enter actual.x.tensor.view.device).call
      (lib.leave ((lib.accessSaved identity).call.run initial).1.owner).call
      (body lib service actual structural dy)) ((lib.accessSaved identity).call.run initial).1

structure SuccessfulCall (lib : APIHost api neg inputs Metadata)
    (service : BackwardRuntime.Service platform api neg inputs Metadata) (identity : Nat)
    (c : SavedContext) (saved : SavedLayout c) (dy : Storage.Tensor) (initial : State Store Metadata) where
  accessed : ((lib.accessSaved identity).call.run initial).2=.ok c
  entered : ((lib.enter c.x.tensor.view.device).call.run ((lib.accessSaved identity).call.run initial).1).2=.ok ()
  body : BodyRoute lib service c saved dy
    ((lib.enter c.x.tensor.view.device).call.run ((lib.accessSaved identity).call.run initial).1).1
  exited : ((lib.leave ((lib.accessSaved identity).call.run initial).1.owner).call.run body.final).2=.ok ()

variable {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {identity : Nat} {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

def SuccessfulCall.accessState (r : SuccessfulCall lib service identity c saved dy initial) :=
  ((lib.accessSaved identity).call.run initial).1

def SuccessfulCall.final (r : SuccessfulCall lib service identity c saved dy initial) :=
  ((lib.leave r.accessState.owner).call.run r.body.final).1

def SuccessfulCall.preTransition (r : SuccessfulCall lib service identity c saved dy initial) :
    Transition api neg inputs initial ((lib.enter c.x.tensor.view.device).call.run r.accessState).1 :=
  ((lib.accessSaved identity).transition initial).then ((lib.enter c.x.tensor.view.device).transition r.accessState)

def SuccessfulCall.realFinal (r : SuccessfulCall lib service identity c saved dy initial)
    (m : Storage.World (Nat → ℝ)) :=
  ((lib.leave r.accessState.owner).transition r.body.final).shadow (r.body.realFinal (r.preTransition.shadow m))

theorem invoke_route (lookup : lib.context initial.metadata identity=some c) (out : GradientReturn.Gradients)
    (ok : (invoke lib service identity c saved dy initial lookup).outcome.result=.ok out) :
    ∃ r : SuccessfulCall lib service identity c saved dy initial,r.body.output=out ∧
      (invoke lib service identity c saved dy initial lookup).outcome.state=r.final := by
  obtain ⟨actual,ha,hd,hs⟩ := CallAt.bind_success _ _ _ ok
  have he := context_identifies lib identity c initial lookup actual ha
  subst actual
  have d := device_success _ _ _ _ _ hd
  obtain ⟨br,hout,hbs⟩ := body_route out d.2.1
  have hl : ((lib.leave ((lib.accessSaved identity).call.run initial).1.owner).call.run br.final).2=.ok () := by
    simpa only [hbs] using d.2.2.1
  let route : SuccessfulCall lib service identity c saved dy initial :=
    ⟨(call_ok _ _ _ _).mp ha,d.1,br,hl⟩
  refine ⟨route,hout,?_⟩
  have hend := d.2.2.2
  rw [hbs] at hend
  exact hs.trans hend

theorem restored_owner (r : SuccessfulCall lib service identity c saved dy initial) : r.final.owner=initial.owner :=
  (lib.leaveOwner r.accessState.owner r.body.final r.exited).trans (lib.accessOwner identity initial)

theorem accessed_live (r : SuccessfulCall lib service identity c saved dy initial)
    (t : Dense) (ht : t∈[c.x,c.w,c.r,c.mu]++c.bias.toList) :
    ((lib.enter c.x.tensor.view.device).call.run r.accessState).1.live t.tensor :=
  (lib.enter c.x.tensor.view.device).transition r.accessState |>.live _
    (lib.accessedPresent identity initial c r.accessed t ht)

theorem all_outcome_frame (lookup : lib.context initial.metadata identity=some c) (key : Key)
    (hk : Storage.caller inputs key) :
    Storage.observe api (invoke lib service identity c saved dy initial lookup).outcome.state.memory key=
      Storage.observe api initial.memory key :=
  (invoke lib service identity c saved dy initial lookup).frame key hk

#axiomsClean invoke_route
#axiomsClean restored_owner
#axiomsClean accessed_live
#axiomsClean all_outcome_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
