import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardReadback
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.GradientReturn
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReceiptAction

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation BackwardLayout ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}

def prepared (lib : APIHost api neg inputs Metadata) (c : SavedContext) (saved : SavedLayout c) (dy : Storage.Tensor) :
    Action (Host.world (Metadata:=Metadata) api inputs) {p : Data // Layout c p inputs} :=
  certify (prepare lib c dy) _ fun s out h => by
    obtain ⟨r,he,_⟩ := prepare_route out h
    subst out
    exact route_layout r saved

def body (lib : APIHost api neg inputs Metadata) (service : BackwardRuntime.Service platform api neg inputs Metadata)
    (c : SavedContext) (saved : SavedLayout c) (dy : Storage.Tensor) :
    Action (Host.world (Metadata:=Metadata) api inputs) GradientReturn.Gradients :=
  (prepared lib c saved dy).bind fun p =>
  (BackwardRuntime.dispatch service c p.val p.property .dx).bind fun _ =>
  (BackwardRuntime.dispatch service c p.val p.property .affine).bind fun _ =>
  GradientReturn.finish lib c p.val

structure BodyRoute (lib : APIHost api neg inputs Metadata)
    (service : BackwardRuntime.Service platform api neg inputs Metadata) (c : SavedContext)
    (saved : SavedLayout c) (dy : Storage.Tensor) (initial : State Store Metadata) where
  preparation : BackwardPreparation.Route lib c dy initial
  dx : BackwardRuntime.Route service c preparation.data (route_layout preparation saved) .dx preparation.final
  affine : BackwardRuntime.Route service c preparation.data (route_layout preparation saved) .affine dx.final
  returns : GradientReturn.Route lib c preparation.data affine.final

variable {lib : APIHost api neg inputs Metadata} {service : BackwardRuntime.Service platform api neg inputs Metadata}
  {c : SavedContext} {saved : SavedLayout c} {dy : Storage.Tensor} {initial : State Store Metadata}

def BodyRoute.output (r : BodyRoute lib service c saved dy initial) := r.returns.output
def BodyRoute.final (r : BodyRoute lib service c saved dy initial) := r.returns.final

def BodyRoute.realFinal (r : BodyRoute lib service c saved dy initial) (m : Storage.World (Nat → ℝ)) :=
  r.returns.transition.shadow (r.affine.realFinal RealProof.ops
    (r.dx.realFinal RealProof.ops (r.preparation.transition.shadow m)))

theorem body_route (out : GradientReturn.Gradients)
    (ok : ((body lib service c saved dy).run initial).result=.ok out) :
    ∃ r : BodyRoute lib service c saved dy initial,r.output=out ∧
      ((body lib service c saved dy).run initial).state=r.final := by
  obtain ⟨p,hp,hd,hsp,_⟩ := bind_success _ _ _ _ ok
  have hprep := certify_success _ _ _ initial p hp
  obtain ⟨pr,he,hstate⟩ := prepare_route p.val hprep
  rcases p with ⟨p,layout⟩
  dsimp only at he
  subst p
  have hpstate : ((prepared lib c saved dy).run initial).state=pr.final :=
    (certify_state _ _ _ initial).trans hstate
  rw [hpstate] at hd hsp
  obtain ⟨unit,hdx,ha,hsx,_⟩ := bind_success _ _ _ _ hd
  cases unit
  obtain ⟨dxr,hdstate⟩ := BackwardRuntime.dispatch_route hdx
  rw [hdstate] at ha hsx
  obtain ⟨unit,haff,hr,hsa,_⟩ := bind_success _ _ _ _ ha
  cases unit
  obtain ⟨ar,hastate⟩ := BackwardRuntime.dispatch_route haff
  rw [hastate] at hr hsa
  obtain ⟨rr,her,hrstate⟩ := GradientReturn.finish_route out hr
  exact ⟨⟨pr,dxr,ar,rr⟩,her,hsp.trans (hsx.trans (hsa.trans hrstate))⟩

theorem epsilon_none (r : BodyRoute lib service c saved dy initial) : r.output.2.2.2=none := rfl

theorem dx_return_storage (r : BodyRoute lib service c saved dy initial) :
    r.output.1.backing=r.preparation.data.dx.tensor.backing ∧ r.output.1.view.shape=c.inputShape := by
  have h := lib.viewed r.preparation.data.dx c.inputShape r.affine.final r.returns.dx r.returns.dxOK
  exact ⟨h.backing,h.shape⟩

theorem dw_return_metadata (r : BodyRoute lib service c saved dy initial) :
    r.output.2.1.view.shape=c.weightShape ∧ r.output.2.1.view.device=c.weightDevice := by
  have h := GradientReturn.transfer_metadata r.returns.wr
  have allocated := (lib.emptyLikeResult c.w r.preparation.sX r.preparation.dw r.preparation.dwOK).2.2
  refine ⟨h.1,?_⟩
  exact h.2.1.trans (r.preparation.data.dw.tensor.owner.trans (allocated.trans saved.weightOwner.symm))

theorem body_frame (lib : APIHost api neg inputs Metadata)
    (service : BackwardRuntime.Service platform api neg inputs Metadata) (c : SavedContext)
    (saved : SavedLayout c) (dy : Storage.Tensor) (s : State Store Metadata) (key : Key)
    (hk : Storage.caller inputs key) :
    Storage.observe api ((body lib service c saved dy).run s).state.memory key=Storage.observe api s.memory key :=
  (body lib service c saved dy).frame s key hk

#axiomsClean body_route
#axiomsClean epsilon_none
#axiomsClean dx_return_storage
#axiomsClean dw_return_metadata
#axiomsClean body_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardCall
