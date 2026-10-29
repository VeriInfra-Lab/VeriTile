import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ForwardReal

namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}

theorem config_state (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x)
    (s : State Store Metadata) : ((config (api:=api) (inputs:=inputs) x w bias shape).run s).state=s := by
  rw [config,certify_state]
  cases h : Compiler.settings shape.columns with
  | error e => simp only [guards,Action.bind,checked,h,Except.mapError]
  | ok cfg =>
    by_cases ha : shape.columns≤w.view.shape.prod ∧ ∀ b,bias=some b → shape.columns≤b.view.shape.prod
    · simp only [guards,Action.bind,checked,check,Action.pure,h,Except.mapError,if_pos ha]
    · simp only [guards,Action.bind,checked,check,Action.pure,h,Except.mapError,if_neg ha]

structure SuccessfulCall (lib : APIHost api neg inputs Metadata) (metadata : MetadataAPI api neg inputs Metadata)
    (service : Runtime.Service platform api neg inputs Metadata) (identity : Nat)
    (x w : Storage.Tensor) (bias : Option Storage.Tensor) (eps : Compiler.Shared.HostScalar)
    (initial : State Store Metadata) where
  shape : ShapeInfo x
  shapeOK : ((metadata.reshapeRows x).call.run initial).2=.ok shape
  (block warps : Nat)
  checks : Checks x w bias shape block warps
  entered : ((lib.enter x.view.device).call.run ((metadata.reshapeRows x).call.run initial).1).2=.ok ()
  body : BodyRoute lib service identity x w bias shape checks eps
    ((lib.enter x.view.device).call.run ((metadata.reshapeRows x).call.run initial).1).1
  exited : ((lib.leave ((metadata.reshapeRows x).call.run initial).1.owner).call.run body.final).2=.ok ()

variable {lib : APIHost api neg inputs Metadata} {metadata : MetadataAPI api neg inputs Metadata}
  {service : Runtime.Service platform api neg inputs Metadata} {identity : Nat}
  {x w : Storage.Tensor} {bias : Option Storage.Tensor} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

def SuccessfulCall.sShape (r : SuccessfulCall lib metadata service identity x w bias eps initial) :=
  ((metadata.reshapeRows x).call.run initial).1

def SuccessfulCall.final (r : SuccessfulCall lib metadata service identity x w bias eps initial) :=
  ((lib.leave r.sShape.owner).call.run r.body.final).1

def SuccessfulCall.preTransition (r : SuccessfulCall lib metadata service identity x w bias eps initial) :
    Transition api neg inputs initial ((lib.enter x.view.device).call.run r.sShape).1 :=
  ((metadata.reshapeRows x).transition initial).then ((lib.enter x.view.device).transition r.sShape)

def SuccessfulCall.realFinal (r : SuccessfulCall lib metadata service identity x w bias eps initial)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) :=
  ((lib.leave r.sShape.owner).transition r.body.final).shadow (r.body.realFinal (r.preTransition.shadow m) epsReal)

theorem forward_route (out : Dense)
    (ok : ((forward lib metadata service identity x w bias eps).run initial).result=.ok out) :
    ∃ r : SuccessfulCall lib metadata service identity x w bias eps initial,r.body.output=out ∧
      ((forward lib metadata service identity x w bias eps).run initial).state=r.final := by
  obtain ⟨shape,hs,rest,hss,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨c,hc,rest2,hsc,_⟩ := bind_success _ _ _ _ rest
  have hstate := config_state (api:=api) (inputs:=inputs) x w bias shape ((metadata.reshapeRows x).call.run initial).1
  dsimp only [call] at rest2 hsc
  rw [hstate] at rest2 hsc
  obtain ⟨old,hget,hdev,hsget,_⟩ := bind_success _ _ _ _ rest2
  have hold : old=((metadata.reshapeRows x).call.run initial).1 := by
    simpa [getState] using hget.symm
  subst old
  have hd := device_success _ _ _ _ _ hdev
  obtain ⟨br,he,hbs⟩ := body_route out hd.2.1
  have hleave : ((lib.leave ((metadata.reshapeRows x).call.run initial).1.owner).call.run br.final).2=.ok () := by
    simpa only [hbs] using hd.2.2.1
  let route : SuccessfulCall lib metadata service identity x w bias eps initial :=
    ⟨shape,(call_ok _ _ _ _).mp hs,c.val.1,c.val.2,c.property,hd.1,br,hleave⟩
  refine ⟨route,he,?_⟩
  have hend := hd.2.2.2
  rw [hbs] at hend
  exact hss.trans (hsc.trans (hsget.trans hend))

theorem output_storage (r : SuccessfulCall lib metadata service identity x w bias eps initial) :
    r.body.output.tensor.backing=r.body.preparation.y.tensor.backing ∧
    r.body.output.tensor.view.shape=x.view.shape ∧ r.body.output.count=x.view.shape.prod :=
  body_output_storage r.body

theorem saved_context (r : SuccessfulCall lib metadata service identity x w bias eps initial) :
    lib.context r.final.metadata identity=some r.body.context := by
  rw [SuccessfulCall.final,lib.leaveContext]
  exact body_context r.body

theorem restored_owner (r : SuccessfulCall lib metadata service identity x w bias eps initial) :
    r.final.owner=initial.owner :=
  (lib.leaveOwner r.sShape.owner r.body.final r.exited).trans (metadata.reshapeOwner x initial)

theorem output_live (r : SuccessfulCall lib metadata service identity x w bias eps initial) :
    r.final.live r.body.output.tensor := by
  have h := (lib.viewed r.body.preparation.data.y x.view.shape r.body.sSave r.body.output r.body.viewed).present
  exact (lib.leave r.sShape.owner).transition r.body.final |>.live _ h

#axiomsClean forward_route
#axiomsClean output_storage
#axiomsClean saved_context
#axiomsClean restored_owner
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
