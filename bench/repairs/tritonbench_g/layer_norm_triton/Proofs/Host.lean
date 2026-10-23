import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Runtime
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReceiptAction
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.GradientReturn

/-! Source host preparation reuses the existing ordinary Torch contracts.
The definitions retain the original physical tensor views and caller backing
spans; every successful materialization is an actual transfer/resolve/clone/view
sequence, including aliases and lazy negation. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation Layout ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

/-- Original Python inputs, with a mandatory affine pair and a normalized
suffix. The shape record includes zero axes; it is not a native-safe subset. -/
def validate (x w b : Storage.Tensor) (normalized : List Nat) :
    Except Failure (Geometry.Shape x.view.shape w.view.shape b.view.shape) :=
  if hn : normalized=[] then .error .rank else
  if hlen : x.view.shape.length<normalized.length then .error .rank else
  if hx : x.view.shape=x.view.shape.take (x.view.shape.length-normalized.length)++normalized then
    if hw : w.view.shape=normalized then
      if hb : b.view.shape=normalized then
        .ok ⟨x.view.shape.take (x.view.shape.length-normalized.length),normalized,hn,hx,hw,hb⟩
      else .error (.assertion (.shape .bias))
    else .error (.assertion (.shape .weight))
  else .error (.assertion (.shape .x))

variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type}

structure Prepared (p : Preparation.Data) (M N : Nat) (inputs : List Storage.Tensor) : Prop where
  nonemptyLayout : 0<N → Layout.Forward p M N inputs
  yCount : p.y.count=M*N

/-- Certificates are extracted from actual successful API calls; they do not
insert new source checks or assume a successful allocation. -/
def prepare (lib : APIHost api neg inputs Metadata) (x w b : Storage.Tensor)
    (shape : Geometry.Shape x.view.shape w.view.shape b.view.shape) :
    Action (VeriTile.Bench.Repairs.FastLayernorm.Host.world (Metadata:=Metadata) api inputs)
      {p : Preparation.Data // Prepared p shape.rows shape.columns inputs} :=
  certify (Preparation.prepare lib x w (some b) shape.rows shape.columns) _ fun s p ok => by
    obtain ⟨r,he,_⟩ := Preparation.prepare_route p ok
    subst p
    have hs := Geometry.shape_product shape
    refine ⟨fun hn => Layout.route_layout r hn hs.1 (by omega) ?_,(Layout.route_counts r hs.1).2.1⟩
    intro original hb
    cases hb
    omega

def forwardRequest (p : Preparation.Data) (layout : Layout.Forward p M N inputs) : Runtime.Request 3 inputs :=
  ⟨Execution.forwardBuffers p layout,[M],
    [(p.x.owner,p.x.address 0),(p.y.owner,p.y.address 0),(p.w.owner,p.w.address 0),
     ((p.bias.getD p.w).owner,(p.bias.getD p.w).address 0),(p.mu.owner,p.mu.address 0),(p.r.owner,p.r.address 0)],
    Layout.reads p⟩

structure ForwardService (platform : Compiler.Platform) (api : Memory.API UInt32 Store UInt8)
    (neg : UInt32 → UInt32) (inputs : List Storage.Tensor) (Metadata : Type) where
  launch : ∀ {M N B : Nat} (p : Preparation.Data) (layout : Layout.Forward p M N inputs)
    (settings : Compiler.settings N=.ok (B,Geometry.warps B)) (eps : Compiler.Shared.HostScalar),
      Runtime.Service platform api neg inputs Metadata .forward M N B 3 eps (forwardRequest p layout)

variable {platform : Compiler.Platform}

def launchForward (service : ForwardService platform api neg inputs Metadata)
    (p : Preparation.Data) (layout : Layout.Forward p M N inputs)
    (settings : Compiler.settings N=.ok (B,Geometry.warps B)) (eps : Compiler.Shared.HostScalar) :=
  Runtime.dispatch (service.launch p layout settings eps) (fun code m =>
    Execution.forwardValues (platform.scalar code.object.primitives) code.plan api p (platform.epsilon code.object eps) m)

def savedContext (identity : Nat) (x w b : Storage.Tensor)
    (shape : Geometry.Shape x.view.shape w.view.shape b.view.shape) (B : Nat)
    (eps : Compiler.Shared.HostScalar) (p : Preparation.Data) : SavedContext :=
  ⟨identity,p.x,p.w,p.r,p.mu,p.bias,x.view.shape,w.view.shape,some b.view.shape,
    w.view.device,some b.view.device,shape.rows,shape.columns,B,Geometry.warps B,eps⟩

def body (lib : APIHost api neg inputs Metadata) (service : ForwardService platform api neg inputs Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (shape : Geometry.Shape x.view.shape w.view.shape b.view.shape)
    (B : Nat) (settings : Compiler.settings shape.columns=.ok (B,Geometry.warps B))
    (eps : Compiler.Shared.HostScalar) :
    Action (VeriTile.Bench.Repairs.FastLayernorm.Host.world (Metadata:=Metadata) api inputs) Dense :=
  (prepare lib x w b shape).bind fun p =>
  ((if active : shape.rows≠0 ∧ shape.columns≠0 then
      launchForward service p.val (p.property.nonemptyLayout (by omega)) settings eps
    else Action.pure ())).bind fun _ =>
  (call .saveOriginal (lib.save (savedContext identity x w b shape B eps p.val)).call).bind fun _ =>
  call .returnTuple (lib.view p.val.y x.view.shape).call

def config (N : Nat) :
    Action (VeriTile.Bench.Repairs.FastLayernorm.Host.world (Metadata:=Metadata) api inputs)
      {B : Nat // Compiler.settings N=.ok (B,Geometry.warps B)} :=
  if h : N>min 16384 N.nextPowerOfTwo then checked .width (.error .width)
  else Action.pure ⟨min 16384 N.nextPowerOfTwo,by simp [Compiler.settings,h]⟩

def forward (lib : APIHost api neg inputs Metadata) (service : ForwardService platform api neg inputs Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (normalized : List Nat) (eps : Compiler.Shared.HostScalar) :
    Action (VeriTile.Bench.Repairs.FastLayernorm.Host.world (Metadata:=Metadata) api inputs) Dense :=
  (checked .unpackShape (validate x w b normalized)).bind fun shape =>
  (config shape.columns).bind fun block =>
  getState.bind fun initial =>
  inDevice (lib.enter x.view.device).call (lib.leave initial.owner).call
    (body lib service identity x w b shape block.val block.property eps)

theorem all_forward_outcomes_frame (lib : APIHost api neg inputs Metadata)
    (service : ForwardService platform api neg inputs Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (normalized : List Nat) (eps : Compiler.Shared.HostScalar)
    (initial : State Store Metadata) (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api ((forward lib service identity x w b normalized eps).run initial).state.memory key=
      Storage.observe api initial.memory key :=
  (forward lib service identity x w b normalized eps).frame initial key hk

/-- This is the already-proved actual host path; arbitrary physical strides,
caller aliases, odd byte tails and lazy-negative views are inherited intact. -/
theorem prepared_values (lib : APIHost api neg inputs Metadata)
    (x w b : Storage.Tensor) (shape : Geometry.Shape x.view.shape w.view.shape b.view.shape)
    (initial : State Store Metadata) (out : {p : Preparation.Data // Prepared p shape.rows shape.columns inputs})
    (ok : ((prepare lib x w b shape).run initial).result=.ok out) :
    ∃ route : Preparation.Route lib x w (some b) shape.rows shape.columns initial,
      route.data=out.val ∧ ((prepare lib x w b shape).run initial).state=route.final := by
  have hp := certify_success _ _ _ initial out ok
  obtain ⟨r,he,hs⟩ := Preparation.prepare_route out.val hp
  exact ⟨r,he,hs⟩

#axiomsClean all_forward_outcomes_frame
#axiomsClean prepared_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
