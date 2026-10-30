import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Readback
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Wrapper
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReceiptAction

/-! Actual forward host path. Certificates attached below are consequences
of the already defined actions. They add neither checks nor failure cases. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Layout Wrapper ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform} {block warps : Nat}

def config (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x) :
    Action (Host.world (Metadata:=Metadata) api inputs) {c : Nat × Nat // Checks x w bias shape c.1 c.2} :=
  certify (guards x w bias shape) _ fun s out h => guards_success x w bias shape s out.1 out.2 h

def prepared (lib : APIHost api neg inputs Metadata) (x w : Storage.Tensor)
    (bias : Option Storage.Tensor) (shape : ShapeInfo x)
    (checks : Checks x w bias shape block warps) :
    Action (Host.world (Metadata:=Metadata) api inputs) {p : Data // Forward p shape.rows shape.columns inputs} :=
  certify (prepare lib x w bias shape.rows shape.columns) _ fun s out h => by
    obtain ⟨r,he,_⟩ := prepare_route out h
    subst out
    exact route_layout r shape.positive (shape_count shape) checks.weight checks.bias

def savedContext (identity : Nat) (x w : Storage.Tensor) (bias : Option Storage.Tensor)
    (shape : ShapeInfo x) (block warps : Nat) (eps : Compiler.Shared.HostScalar) (p : Data) : SavedContext :=
  ⟨identity,p.x,p.w,p.r,p.mu,p.bias,x.view.shape,w.view.shape,bias.map (fun b => b.view.shape),
    w.view.device,bias.map (fun b => b.view.device),shape.rows,shape.columns,block,warps,eps⟩

def settingEquation (checks : Checks x w bias shape block warps) :
    Compiler.settings shape.columns=.ok (block,Compiler.warps block) := by
  have h := (Compiler.settings_success _ _ _ checks.settings).2.2.2
  simpa only [h] using checks.settings

def body (lib : APIHost api neg inputs Metadata) (service : Runtime.Service platform api neg inputs Metadata)
    (identity : Nat) (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x)
    (checks : Checks x w bias shape block warps) (eps : Compiler.Shared.HostScalar) :
    Action (Host.world (Metadata:=Metadata) api inputs) Dense :=
  (prepared lib x w bias shape checks).bind fun p =>
  (Runtime.dispatch service p.val p.property (settingEquation checks) eps).bind fun _ =>
  (call .saveOriginal (lib.save (savedContext identity x w bias shape block warps eps p.val)).call).bind fun _ =>
  call .returnTuple (lib.view p.val.y x.view.shape).call

def forward (lib : APIHost api neg inputs Metadata) (metadata : MetadataAPI api neg inputs Metadata)
    (service : Runtime.Service platform api neg inputs Metadata) (identity : Nat)
    (x w : Storage.Tensor) (bias : Option Storage.Tensor) (eps : Compiler.Shared.HostScalar) :
    Action (Host.world (Metadata:=Metadata) api inputs) Dense :=
  (call .unpackShape (metadata.reshapeRows x).call).bind fun shape =>
  (config x w bias shape).bind fun c =>
  getState.bind fun initial =>
  inDevice (lib.enter x.view.device).call (lib.leave initial.owner).call
    (body lib service identity x w bias shape c.property eps)

def publicForward (lib : APIHost api neg inputs Metadata) (metadata : MetadataAPI api neg inputs Metadata)
    (service : Runtime.Service platform api neg inputs Metadata) (identity : Nat) (module : Module)
    (x : Storage.Tensor) : Action (Host.world (Metadata:=Metadata) api inputs) Dense :=
  (check (.shape .weight) (module.affine=Bool.true) (.assertion (.shape .weight))).bind fun _ =>
  forward lib metadata service identity x module.weight module.bias module.selectedEpsilon

structure BodyRoute (lib : APIHost api neg inputs Metadata)
    (service : Runtime.Service platform api neg inputs Metadata) (identity : Nat)
    (x w : Storage.Tensor) (bias : Option Storage.Tensor) (shape : ShapeInfo x)
    (checks : Checks x w bias shape block warps) (eps : Compiler.Shared.HostScalar)
    (initial : State Store Metadata) where
  preparation : Preparation.Route lib x w bias shape.rows shape.columns initial
  dispatch : Runtime.Route service preparation.data
    (route_layout preparation shape.positive (shape_count shape) checks.weight checks.bias)
    (settingEquation checks) eps preparation.final
  saved : ((lib.save (savedContext identity x w bias shape block warps eps preparation.data)).call.run dispatch.final).2=.ok ()
  output : Dense
  viewed : ((lib.view preparation.data.y x.view.shape).call.run
    ((lib.save (savedContext identity x w bias shape block warps eps preparation.data)).call.run dispatch.final).1).2=.ok output

variable {lib : APIHost api neg inputs Metadata} {service : Runtime.Service platform api neg inputs Metadata}
  {identity : Nat} {x w : Storage.Tensor} {bias : Option Storage.Tensor} {shape : ShapeInfo x}
  {block warps : Nat} {checks : Checks x w bias shape block warps} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

def BodyRoute.context (r : BodyRoute lib service identity x w bias shape checks eps initial) :=
  savedContext identity x w bias shape block warps eps r.preparation.data

def BodyRoute.sSave (r : BodyRoute lib service identity x w bias shape checks eps initial) :=
  ((lib.save r.context).call.run r.dispatch.final).1

def BodyRoute.final (r : BodyRoute lib service identity x w bias shape checks eps initial) :=
  ((lib.view r.preparation.data.y x.view.shape).call.run r.sSave).1

def BodyRoute.tail (r : BodyRoute lib service identity x w bias shape checks eps initial) :
    Transition api neg inputs r.dispatch.final r.final :=
  ((lib.save r.context).transition r.dispatch.final).then
    ((lib.view r.preparation.data.y x.view.shape).transition r.sSave)

theorem body_route (out : Dense)
    (ok : ((body lib service identity x w bias shape checks eps).run initial).result=.ok out) :
    ∃ r : BodyRoute lib service identity x w bias shape checks eps initial,r.output=out ∧
      ((body lib service identity x w bias shape checks eps).run initial).state=r.final := by
  obtain ⟨p,hp,hd,hsp,_⟩ := bind_success _ _ _ _ ok
  have hprep := certify_success _ _ _ initial p hp
  obtain ⟨pr,he,hstate⟩ := prepare_route p.val hprep
  rcases p with ⟨p,layout⟩
  dsimp only at he
  subst p
  have hpstate : ((prepared lib x w bias shape checks).run initial).state=pr.final :=
    (certify_state _ _ _ initial).trans hstate
  rw [hpstate] at hd hsp
  obtain ⟨ignored,hdispatch,hpost,hsd,_⟩ := bind_success _ _ _ _ hd
  obtain ⟨dr,hignored,hdstate⟩ := Runtime.dispatch_route ignored hdispatch
  rw [hdstate] at hpost hsd
  obtain ⟨unit,hsave,hview,hss,_⟩ := bind_success _ _ _ _ hpost
  cases unit
  have hs := (call_ok _ _ _ _).mp hsave
  have hv := (call_ok _ _ _ _).mp hview
  exact ⟨⟨pr,dr,hs,out,hv⟩,rfl,hsp.trans (hsd.trans hss)⟩

theorem body_output_storage (r : BodyRoute lib service identity x w bias shape checks eps initial) :
    r.output.tensor.backing=r.preparation.y.tensor.backing ∧ r.output.tensor.view.shape=x.view.shape ∧
    r.output.count=x.view.shape.prod := by
  have hv := lib.viewed r.preparation.data.y x.view.shape r.sSave r.output r.viewed
  refine ⟨hv.backing,hv.shape,?_⟩
  rw [r.output.countEq,hv.shape]

theorem body_context (r : BodyRoute lib service identity x w bias shape checks eps initial) :
    lib.context r.final.metadata identity=some r.context := by
  rw [BodyRoute.final,lib.viewContext]
  exact lib.saved r.context r.dispatch.final r.saved

theorem body_output_live (r : BodyRoute lib service identity x w bias shape checks eps initial) :
    r.dispatch.final.live r.preparation.data.y.tensor := by
  exact r.dispatch.preTransition.live _ ((route_present r.preparation).2 (2 : Fin 3))

theorem body_native_readback (r : BodyRoute lib service identity x w bias shape checks eps initial)
    (i : Fin r.preparation.data.y.count) :
    Storage.logical api neg r.final.memory r.output.tensor i.val=
      forwardReference (platform.scalar r.dispatch.code.object.primitives) r.dispatch.code.plan
        (Execution.rows r.preparation.data.x shape.columns r.dispatch.s2.memory api (i.val/shape.columns))
        (Execution.vector r.preparation.data.w r.dispatch.s2.memory api)
        (Runtime.biasValues (platform.scalar r.dispatch.code.object.primitives) api r.preparation.data r.dispatch.s2.memory)
        (platform.epsilon r.dispatch.code.object eps) (i.val%shape.columns) := by
  have hv := lib.viewed r.preparation.data.y x.view.shape r.sSave r.output r.viewed
  have hy := body_output_live r
  have hi : i.val<r.preparation.data.y.tensor.view.shape.prod := by
    rw [←r.preparation.data.y.countEq]; exact i.isLt
  have hview := hv.values i.val i.isLt
  change Storage.logical api neg r.final.memory r.output.tensor i.val=
    Storage.logical api neg r.sSave.memory r.preparation.data.y.tensor i.val at hview
  have hsave := (lib.save r.context).transition r.dispatch.final |>.values _ hy i.val hi
  change Storage.logical api neg r.sSave.memory r.preparation.data.y.tensor i.val=
    Storage.logical api neg r.dispatch.final.memory r.preparation.data.y.tensor i.val at hsave
  rw [hview,hsave,Storage.dense_logical api neg _ _ i.val i.isLt]
  have ho := route_output_owners r.preparation (2 : Fin 3)
  change r.preparation.data.y.owner=r.preparation.data.x.owner at ho
  rw [ho]
  exact Runtime.route_y_readback r.dispatch i

/-- Caller memory is the complete union of the supplied backing stores. This
holds on setup, compilation, execution, save, view and context-exit failures. -/
theorem forward_frame (lib : APIHost api neg inputs Metadata) (metadata : MetadataAPI api neg inputs Metadata)
    (service : Runtime.Service platform api neg inputs Metadata) (identity : Nat)
    (x w : Storage.Tensor) (bias : Option Storage.Tensor) (eps : Compiler.Shared.HostScalar)
    (s : State Store Metadata) (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api ((forward lib metadata service identity x w bias eps).run s).state.memory key=
      Storage.observe api s.memory key :=
  (forward lib metadata service identity x w bias eps).frame s key hk

#axiomsClean body_route
#axiomsClean body_output_storage
#axiomsClean body_context
#axiomsClean body_native_readback
#axiomsClean forward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
