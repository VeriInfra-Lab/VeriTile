import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Scratch

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor} {Metadata : Type}

def groupRequest (c : SavedContext) (base : Base) (scratch : Scratch) (layout : BaseLayout base c inputs)
    (sl : ScratchLayout base scratch c inputs) : Runtime.Request 3 inputs :=
  ⟨(groupBuffers base scratch layout sl).buffers,[min (Geometry.groups c.columns) c.rows],
    [(base.dx.owner,base.dx.address 0),(base.dy.owner,base.dy.address 0),
     (scratch.dw.owner,scratch.dw.address 0),(scratch.db.owner,scratch.db.address 0),
     (c.x.owner,c.x.address 0),(c.w.owner,c.w.address 0),(c.mu.owner,c.mu.address 0),(c.r.owner,c.r.address 0)],
    [c.x,c.w,base.dy,c.mu,c.r]⟩

def finalRequest (c : SavedContext) (base : Base) (scratch : Scratch) (layout : BaseLayout base c inputs) :
    Runtime.Request 2 inputs :=
  ⟨(finalBuffers base layout).buffers,[(c.columns+127)/128],
    [(scratch.dw.owner,scratch.dw.address 0),(scratch.db.owner,scratch.db.address 0),
     (base.dw.owner,base.dw.address 0),(base.db.owner,base.db.address 0)],
    [scratch.dw,scratch.db]⟩

structure Service (platform : Compiler.Platform) (api : Memory.API UInt32 Store UInt8)
    (neg : UInt32 → UInt32) (inputs : List Storage.Tensor) (Metadata : Type) where
  first : ∀ c (base : Base) (scratch : Scratch) (layout : BaseLayout base c inputs)
    (sl : ScratchLayout base scratch c inputs),
      Runtime.Service platform api neg inputs Metadata .groupedBackward c.rows c.columns c.block 3 c.epsilon
        (groupRequest c base scratch layout sl)
  last : ∀ c (base : Base) (scratch : Scratch) (layout : BaseLayout base c inputs),
      Runtime.Service platform api neg inputs Metadata .finalBackward c.rows c.columns c.block 2 c.epsilon
        (finalRequest c base scratch layout)

variable {platform : Compiler.Platform}

def launches (service : Service platform api neg inputs Metadata) (c : SavedContext) (base : Base) (scratch : Scratch)
    (layout : BaseLayout base c inputs) (sl : ScratchLayout base scratch c inputs) :
    Action (world (Metadata:=Metadata) api inputs) Unit :=
  (Runtime.dispatch (service.first c base scratch layout sl) (fun code m =>
    Execution.groupValues (platform.scalar code.object.primitives) code.plan api
      (groupBuffers base scratch layout sl) c.x c.w base.dy c.r c.mu m)).bind fun _ =>
  Runtime.dispatch (service.last c base scratch layout) (fun code m =>
    Execution.finalValues (platform.scalar code.object.primitives) code.finalPlan api
      (finalBuffers base layout) (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db m)

/-- Exactly five autograd slots: dX, None, dWeight, dBias, None. -/
def finish (lib : APIHost api neg inputs Metadata) (c : SavedContext) (p : Base) :
    Action (world (Metadata:=Metadata) api inputs) (List (Option Storage.Tensor)) :=
  (call .returnTuple (lib.view p.dx c.inputShape).call).bind fun dx =>
  (GradientReturn.transferView lib p.dw c.weightShape c.weightDevice).bind fun dw =>
  (GradientReturn.transferView lib p.db (c.biasShape.getD []) (c.biasDevice.getD 0)).bind fun db =>
  Action.pure [some dx.tensor,none,some dw,some db,none]

def launchPrepared (service : Service platform api neg inputs Metadata) (c : SavedContext)
    (data : All) (ready : Ready data c inputs) : Action (world (Metadata:=Metadata) api inputs) Unit :=
  match hs : data.scratch with
  | none => Action.pure ()
  | some scratch => launches service c data.base scratch ready.base (ready.scratch scratch hs)

def body (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (service : Service platform api neg inputs Metadata) (c : SavedContext) (b : Dense) (saved : Saved c b)
    (dy : Storage.Tensor) : Action (world (Metadata:=Metadata) api inputs) (List (Option Storage.Tensor)) :=
  (prepared lib extra c b saved dy).bind fun p =>
  (launchPrepared service c p.val p.property).bind fun _ =>
  finish lib c p.val.base

structure Invocation (api : Memory.API UInt32 Store UInt8) (inputs : List Storage.Tensor)
    (Metadata : Type) (initial : State Store Metadata) where
  outcome : Outcome (State Store Metadata) (List (Option Storage.Tensor))
  frame : ∀ key,Storage.caller inputs key → Storage.observe api outcome.state.memory key=Storage.observe api initial.memory key

/-- The actual saved-tensor accessor runs first, including version failure.
A successful result is identified with the context already saved in metadata;
no equality of saved statistics to a mathematical formula is assumed here. -/
def backward (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (service : Service platform api neg inputs Metadata) (identity : Nat) (c : SavedContext)
    (b : Dense) (saved : Saved c b) (dy : Storage.Tensor) (initial : State Store Metadata)
    (lookup : lib.context initial.metadata identity=some c) : Invocation api inputs Metadata initial :=
  let access := call .saveOriginal (lib.accessSaved identity).call
  let accessed := access.run initial
  match h : accessed.result with
  | .error e => ⟨⟨accessed.state,.error e,accessed.trace⟩,access.frame initial⟩
  | .ok actual =>
    have ha : ((lib.accessSaved identity).call.run initial).2=.ok actual := (call_ok _ _ _ _).mp h
    have _same : actual=c := Option.some.inj ((lib.accessed identity initial actual ha).symm.trans lookup)
    let action := inDevice (lib.enter c.x.tensor.view.device).call (lib.leave accessed.state.owner).call
      (body lib extra service c b saved dy)
    ⟨action.run accessed.state,fun key hk => (action.frame accessed.state key hk).trans (access.frame initial key hk)⟩

structure ReturnRoute (lib : APIHost api neg inputs Metadata) (c : SavedContext) (p : Base)
    (initial : State Store Metadata) where
  dx : Dense
  dxOK : ((lib.view p.dx c.inputShape).call.run initial).2=.ok dx
  wr : GradientReturn.TransferRoute lib p.dw c.weightShape c.weightDevice ((lib.view p.dx c.inputShape).call.run initial).1
  br : GradientReturn.TransferRoute lib p.db (c.biasShape.getD []) (c.biasDevice.getD 0) wr.final

variable {lib : APIHost api neg inputs Metadata} {c : SavedContext} {p : Base} {initial : State Store Metadata}
def ReturnRoute.output (r : ReturnRoute lib c p initial) : List (Option Storage.Tensor) :=
  [some r.dx.tensor,none,some r.wr.output,some r.br.output,none]
def ReturnRoute.final (r : ReturnRoute lib c p initial) := r.br.final

theorem finish_route (out : List (Option Storage.Tensor)) (ok : ((finish lib c p).run initial).result=.ok out) :
    ∃ r : ReturnRoute lib c p initial,r.output=out ∧ ((finish lib c p).run initial).state=r.final := by
  obtain ⟨dx,hdx,hw,hsx,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨dw,hdw,hb,hsw,_⟩ := bind_success _ _ _ _ hw
  obtain ⟨wr,he,hswr⟩ := GradientReturn.transfer_route dw hdw
  subst dw
  rw [hswr] at hb hsw
  obtain ⟨db,hdb,hret,hsb,_⟩ := bind_success _ _ _ _ hb
  obtain ⟨br,he,hsbr⟩ := GradientReturn.transfer_route db hdb
  subst db
  let r : ReturnRoute lib c p initial := ⟨dx,(call_ok _ _ _ _).mp hdx,wr,br⟩
  have hout : out=r.output := by simpa [r,ReturnRoute.output,Action.pure] using hret.symm
  subst out
  exact ⟨r,rfl,hsx.trans (hsw.trans (hsb.trans hsbr))⟩

theorem returned_metadata (r : ReturnRoute lib c p initial) :
    r.dx.tensor.backing=p.dx.tensor.backing ∧ r.dx.tensor.view.shape=c.inputShape ∧
    r.wr.output.view.shape=c.weightShape ∧ r.wr.output.view.device=p.dw.tensor.view.device ∧
    r.br.output.view.shape=c.biasShape.getD [] ∧ r.br.output.view.device=p.db.tensor.view.device := by
  have hx := lib.viewed p.dx c.inputShape initial r.dx r.dxOK
  have hw := GradientReturn.transfer_metadata r.wr
  have hb := GradientReturn.transfer_metadata r.br
  exact ⟨hx.backing,hx.shape,hw.1,hw.2.1,hb.1,hb.2.1⟩

#axiomsClean finish_route
#axiomsClean returned_metadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
