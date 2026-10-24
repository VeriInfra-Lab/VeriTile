import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Host
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.StoredReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation Layout ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}

inductive LaunchRoute (service : ForwardService platform api neg inputs Metadata)
    (p : Preparation.Data) (layout : Prepared p M N inputs)
    (settings : Compiler.settings N=.ok (B,Geometry.warps B)) (eps : Compiler.Shared.HostScalar)
    (initial : State Store Metadata) where
  | skipped (inactive : ¬(M≠0 ∧ N≠0))
  | ran (active : M≠0 ∧ N≠0)
      (route : Runtime.Route (service.launch p (layout.nonemptyLayout (by omega)) settings eps)
        (fun code m => Execution.forwardValues (platform.scalar code.object.primitives) code.plan api p
          (platform.epsilon code.object eps) m) initial)

variable {service : ForwardService platform api neg inputs Metadata} {p : Preparation.Data}
  {layout : Prepared p M N inputs} {settings : Compiler.settings N=.ok (B,Geometry.warps B)}
  {eps : Compiler.Shared.HostScalar} {initial : State Store Metadata}

def LaunchRoute.final : LaunchRoute service p layout settings eps initial → State Store Metadata
  | .skipped _ => initial | .ran _ r => r.final

def LaunchRoute.realFinal (r : LaunchRoute service p layout settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) : Storage.World (Nat → ℝ) :=
  match r with
  | .skipped _ => m
  | .ran active r =>
      let before := r.preTransition.shadow m
      (Execution.forwardBuffers p (layout.nonemptyLayout (by omega))).run Memory.realAPI
        (Execution.forwardValues RealProof.ops r.code.plan Memory.realAPI p epsilon before) r.schedule before

theorem launch_route (ok : ((if active : M≠0 ∧ N≠0 then
    launchForward service p (layout.nonemptyLayout (by omega)) settings eps else Action.pure ()).run initial).result=.ok ()) :
    ∃ r : LaunchRoute service p layout settings eps initial,
      ((if active : M≠0 ∧ N≠0 then launchForward service p (layout.nonemptyLayout (by omega)) settings eps
        else Action.pure ()).run initial).state=r.final := by
  split at ok
  · rename_i active
    obtain ⟨r,hs⟩ := Runtime.dispatch_route ok
    exact ⟨.ran active r,by simpa [active,LaunchRoute.final,launchForward] using hs⟩
  · rename_i inactive
    exact ⟨.skipped inactive,by simp [inactive,Action.pure,LaunchRoute.final]⟩

theorem launch_live (r : LaunchRoute service p layout settings eps initial) (t : Storage.Tensor)
    (present : initial.live t) : r.final.live t := by
  cases r with
  | skipped _ => exact present
  | ran _ r => exact r.preTransition.live t present

structure BodyRoute (lib : APIHost api neg inputs Metadata) (service : ForwardService platform api neg inputs Metadata)
    (identity : Nat) (x w b : Storage.Tensor) (shape : Geometry.Shape x.view.shape w.view.shape b.view.shape)
    (B : Nat) (settings : Compiler.settings shape.columns=.ok (B,Geometry.warps B))
    (eps : Compiler.Shared.HostScalar) (initial : State Store Metadata) where
  preparation : Preparation.Route lib x w (some b) shape.rows shape.columns initial
  layout : Prepared preparation.data shape.rows shape.columns inputs
  launch : LaunchRoute service preparation.data layout settings eps preparation.final
  saved : ((lib.save (savedContext identity x w b shape B eps preparation.data)).call.run launch.final).2=.ok ()
  output : Dense
  viewed : ((lib.view preparation.data.y x.view.shape).call.run
    ((lib.save (savedContext identity x w b shape B eps preparation.data)).call.run launch.final).1).2=.ok output

variable {lib : APIHost api neg inputs Metadata} {identity : Nat} {x w b : Storage.Tensor}
  {shape : Geometry.Shape x.view.shape w.view.shape b.view.shape} {B : Nat}
  {settings : Compiler.settings shape.columns=.ok (B,Geometry.warps B)}

def BodyRoute.context (r : BodyRoute lib service identity x w b shape B settings eps initial) :=
  savedContext identity x w b shape B eps r.preparation.data

def BodyRoute.sSave (r : BodyRoute lib service identity x w b shape B settings eps initial) :=
  ((lib.save r.context).call.run r.launch.final).1

def BodyRoute.final (r : BodyRoute lib service identity x w b shape B settings eps initial) :=
  ((lib.view r.preparation.data.y x.view.shape).call.run r.sSave).1

def BodyRoute.tail (r : BodyRoute lib service identity x w b shape B settings eps initial) :
    Transition api neg inputs r.launch.final r.final :=
  ((lib.save r.context).transition r.launch.final).then ((lib.view r.preparation.data.y x.view.shape).transition r.sSave)

def BodyRoute.realFinal (r : BodyRoute lib service identity x w b shape B settings eps initial)
    (m : Storage.World (Nat → ℝ)) (epsilon : ℝ) :=
  r.tail.shadow (r.launch.realFinal (r.preparation.transition.shadow m) epsilon)

theorem body_route (out : Dense)
    (ok : ((body lib service identity x w b shape B settings eps).run initial).result=.ok out) :
    ∃ r : BodyRoute lib service identity x w b shape B settings eps initial,
      r.output=out ∧ ((body lib service identity x w b shape B settings eps).run initial).state=r.final := by
  obtain ⟨p,hp,hd,hsp,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨pr,he,hstate⟩ := prepared_values lib x w b shape initial p hp
  rcases p with ⟨p,layout⟩
  dsimp only at he
  subst p
  rw [hstate] at hd hsp
  obtain ⟨unit,hl,hsave,hsl,_⟩ := bind_success _ _ _ _ hd
  cases unit
  obtain ⟨lr,hlstate⟩ := launch_route hl
  rw [hlstate] at hsave hsl
  obtain ⟨unit,hsv,hview,hss,_⟩ := bind_success _ _ _ _ hsave
  cases unit
  exact ⟨⟨pr,layout,lr,(call_ok _ _ _ _).mp hsv,out,(call_ok _ _ _ _).mp hview⟩,rfl,
    hsp.trans (hsl.trans hss)⟩

theorem body_storage (r : BodyRoute lib service identity x w b shape B settings eps initial) :
    r.output.tensor.backing=r.preparation.data.y.tensor.backing ∧ r.output.tensor.view.shape=x.view.shape := by
  have hv := lib.viewed r.preparation.data.y x.view.shape r.sSave r.output r.viewed
  exact ⟨hv.backing,hv.shape⟩

theorem body_saved_context (r : BodyRoute lib service identity x w b shape B settings eps initial) :
    lib.context r.final.metadata identity=some r.context := by
  rw [BodyRoute.final,lib.viewContext]
  exact lib.saved r.context r.launch.final r.saved

#axiomsClean launch_route
#axiomsClean body_route
#axiomsClean body_storage
#axiomsClean body_saved_context
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
