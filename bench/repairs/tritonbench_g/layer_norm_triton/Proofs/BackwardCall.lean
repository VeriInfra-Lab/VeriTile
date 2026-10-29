import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardLogical
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReturnShape

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}

structure Call (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (service : Service platform api neg inputs Metadata) (identity : Nat) (c : SavedContext) (b : Dense)
    (saved : Saved c b) (dy : Storage.Tensor) (initial : State Store Metadata) where
  accessed : ((lib.accessSaved identity).call.run initial).2=.ok c
  entered : ((lib.enter c.x.tensor.view.device).call.run ((lib.accessSaved identity).call.run initial).1).2=.ok ()
  body : BodyRoute lib extra service c b saved dy
    ((lib.enter c.x.tensor.view.device).call.run ((lib.accessSaved identity).call.run initial).1).1
  exited : ((lib.leave ((lib.accessSaved identity).call.run initial).1.owner).call.run body.final).2=.ok ()

variable {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata}
  {service : Service platform api neg inputs Metadata} {identity : Nat}
  {c : SavedContext} {b : Dense} {saved : Saved c b} {dy : Storage.Tensor} {initial : State Store Metadata}

def Call.accessState (r : Call lib extra service identity c b saved dy initial) := ((lib.accessSaved identity).call.run initial).1

def Call.final (r : Call lib extra service identity c b saved dy initial) :=
  ((lib.leave r.accessState.owner).call.run r.body.final).1

def Call.preTransition (r : Call lib extra service identity c b saved dy initial) :
    Transition api neg inputs initial ((lib.enter c.x.tensor.view.device).call.run r.accessState).1 :=
  ((lib.accessSaved identity).transition initial).then ((lib.enter c.x.tensor.view.device).transition r.accessState)

def Call.realFinal (r : Call lib extra service identity c b saved dy initial) (m : Storage.World (Nat → ℝ)) :=
  ((lib.leave r.accessState.owner).transition r.body.final).shadow (r.body.realFinal (r.preTransition.shadow m))

theorem public_route (lookup : lib.context initial.metadata identity=some c) (out : List (Option Storage.Tensor))
    (ok : (backward lib extra service identity c b saved dy initial lookup).outcome.result=.ok out) :
    ∃ r : Call lib extra service identity c b saved dy initial,r.body.output=out ∧
      (backward lib extra service identity c b saved dy initial lookup).outcome.state=r.final := by
  unfold backward at ok
  dsimp only at ok
  split at ok
  · cases ok
  · rename_i actual ha
    have hac : ((lib.accessSaved identity).call.run initial).2=.ok actual := (call_ok _ _ _ _).mp ha
    have he : actual=c := Option.some.inj ((lib.accessed identity initial actual hac).symm.trans lookup)
    subst actual
    obtain ⟨entered,hbody,left,hstate⟩ := device_success _ _ _ _ out ok
    obtain ⟨br,hout,hbs⟩ := body_route out hbody
    rw [hbs] at left hstate
    let r : Call lib extra service identity c b saved dy initial := ⟨hac,entered,br,left⟩
    refine ⟨r,hout,?_⟩
    unfold backward
    dsimp only
    split
    · rename_i e herr
      have bad := ha.symm.trans herr
      cases bad
    · exact hstate

theorem restored_owner (r : Call lib extra service identity c b saved dy initial) : r.final.owner=initial.owner :=
  (lib.leaveOwner r.accessState.owner r.body.final r.exited).trans (lib.accessOwner identity initial)

theorem accessed_live (r : Call lib extra service identity c b saved dy initial)
    (t : Dense) (ht : t∈[c.x,c.w,c.r,c.mu]++c.bias.toList) :
    ((lib.enter c.x.tensor.view.device).call.run r.accessState).1.live t.tensor :=
  (lib.enter c.x.tensor.view.device).transition r.accessState |>.live _
    (lib.accessedPresent identity initial c r.accessed t ht)

def ReturnRoute.outputs (r : ReturnRoute lib c p initial) : Fin 3 → Storage.Tensor
  | ⟨0,_⟩ => r.dx.tensor | ⟨1,_⟩ => r.wr.output | ⟨2,_⟩ => r.br.output

theorem return_present (r : ReturnRoute lib c p initial) (k : Fin 3) : r.final.live (r.outputs k) := by
  fin_cases k
  · exact r.br.transition.live _ (r.wr.transition.live _ (lib.viewed p.dx c.inputShape initial r.dx r.dxOK).present)
  · exact r.br.transition.live _ (GradientReturn.transfer_metadata r.wr).2.2
  · exact (GradientReturn.transfer_metadata r.br).2.2

theorem return_count (r : ReturnRoute lib c p initial) (k : Fin 3) :
    (r.outputs k).view.shape.prod=(outputs p k).count := by
  fin_cases k
  · have h := lib.viewed p.dx c.inputShape initial r.dx r.dxOK
    exact r.dx.countEq.symm.trans h.count
  · change r.wr.output.view.shape.prod=p.dw.count
    have h := lib.viewed p.dw c.weightShape ((lib.view p.dx c.inputShape).call.run initial).1 r.wr.viewed r.wr.viewOK
    rw [(GradientReturn.transfer_metadata r.wr).1,←h.shape,←r.wr.viewed.countEq,h.count]
  · exact (GradientReturn.db_metadata r.reuse p.db r.br.output rfl rfl).2.2

theorem post_native (r : Call lib extra service identity c b saved dy initial)
    (k : Fin 3) (i : Fin (outputs r.body.preparation.data.base k).count) :
    Storage.logical api neg r.final.memory (r.body.returns.outputs k) i.val=
      Storage.logical api neg r.body.final.memory (r.body.returns.outputs k) i.val :=
  (lib.leave r.accessState.owner).transition r.body.final |>.values _ (return_present r.body.returns k) i.val
    (by rw [return_count]; exact i.isLt)

theorem post_real (r : Call lib extra service identity c b saved dy initial) (m : Storage.World (Nat → ℝ))
    (k : Fin 3) (i : Fin (outputs r.body.preparation.data.base k).count) :
    Storage.realLogical (r.realFinal m) (r.body.returns.outputs k) i.val=
      Storage.realLogical (r.body.realFinal (r.preTransition.shadow m)) (r.body.returns.outputs k) i.val :=
  (lib.leave r.accessState.owner).transition r.body.final |>.realValues _ _ (return_present r.body.returns k) i.val
    (by rw [return_count]; exact i.isLt)

theorem returned_native (r : Call lib extra service identity c b saved dy initial)
    (k : Fin 3) (i : Fin (outputs r.body.preparation.data.base k).count) :
    Storage.logical api neg r.final.memory (r.body.returns.outputs k) i.val=r.body.launch.nativeValue k i.val := by
  apply (post_native r k i).trans
  fin_cases k
  · exact (body_native r.body).1 i
  · exact (body_native r.body).2.1 i
  · exact (body_native r.body).2.2 i

theorem all_outcome_frame (lookup : lib.context initial.metadata identity=some c) (key : Key)
    (hk : Storage.caller inputs key) :
    Storage.observe api (backward lib extra service identity c b saved dy initial lookup).outcome.state.memory key=
      Storage.observe api initial.memory key := (backward lib extra service identity c b saved dy initial lookup).frame key hk

#axiomsClean public_route
#axiomsClean restored_owner
#axiomsClean returned_native
#axiomsClean all_outcome_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
