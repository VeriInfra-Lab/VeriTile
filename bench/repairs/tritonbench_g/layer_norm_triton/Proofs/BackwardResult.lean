import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Backward
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.StoredReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}

inductive ScratchBranch (extra : APIExtra api neg inputs Metadata) (c : SavedContext) (initial : State Store Metadata) where
  | skipped (inactive : ¬(c.rows≠0 ∧ c.columns≠0))
  | allocated (active : c.rows≠0 ∧ c.columns≠0) (route : ScratchRoute extra c initial)

def ScratchBranch.output {extra : APIExtra api neg inputs Metadata} {c : SavedContext} {initial : State Store Metadata} :
    ScratchBranch extra c initial → Option Scratch
  | .skipped _ => none | .allocated _ r => some r.data

def ScratchBranch.final {extra : APIExtra api neg inputs Metadata} {c : SavedContext} {initial : State Store Metadata} :
    ScratchBranch extra c initial → State Store Metadata
  | .skipped _ => initial | .allocated _ r => r.final

def ScratchBranch.transition {extra : APIExtra api neg inputs Metadata} {c : SavedContext} {initial : State Store Metadata}
    (r : ScratchBranch extra c initial) : Transition api neg inputs initial r.final :=
  match r with | .skipped _ => .refl initial | .allocated _ r => r.transition

structure PreparationRoute (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (c : SavedContext) (b : Dense) (dy : Storage.Tensor) (initial : State Store Metadata) where
  base : BaseRoute lib extra c b dy initial
  scratch : ScratchBranch extra c base.final

variable {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata}
  {c : SavedContext} {b : Dense} {dy : Storage.Tensor} {initial : State Store Metadata}

def PreparationRoute.data (r : PreparationRoute lib extra c b dy initial) : All := ⟨r.base.data,r.scratch.output⟩
def PreparationRoute.final (r : PreparationRoute lib extra c b dy initial) := r.scratch.final

def PreparationRoute.transition (r : PreparationRoute lib extra c b dy initial) : Transition api neg inputs initial r.final :=
  r.base.transition.then r.scratch.transition

theorem preparation_route (out : All) (ok : ((prepareAll lib extra c b dy).run initial).result=.ok out) :
    ∃ r : PreparationRoute lib extra c b dy initial,r.data=out ∧
      ((prepareAll lib extra c b dy).run initial).state=r.final := by
  obtain ⟨base,hb,rest,hs,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨br,he,hstate⟩ := base_route base hb
  subst base
  rw [hstate] at rest hs
  split at rest
  · rename_i active
    obtain ⟨s,hsc,hret,hss,_⟩ := bind_success _ _ _ _ rest
    obtain ⟨sr,he,hscstate⟩ := scratch_route s hsc
    subst s
    let r : PreparationRoute lib extra c b dy initial := ⟨br,.allocated active sr⟩
    have hout : out=r.data := by simpa [r,PreparationRoute.data,ScratchBranch.output,Action.pure] using hret.symm
    subst out
    refine ⟨r,rfl,hs.trans ?_⟩
    simpa [active,prepareAll] using hss.trans hscstate
  · rename_i inactive
    let r : PreparationRoute lib extra c b dy initial := ⟨br,.skipped inactive⟩
    have hout : out=r.data := by simpa [r,PreparationRoute.data,ScratchBranch.output,Action.pure] using rest.symm
    subst out
    exact ⟨r,rfl,by simpa [prepareAll,inactive,Action.pure,r,PreparationRoute.final,ScratchBranch.final] using hs⟩

structure LaunchesRoute (service : Service platform api neg inputs Metadata) (c : SavedContext)
    (base : Base) (scratch : Scratch) (layout : BaseLayout base c inputs) (sl : ScratchLayout base scratch c inputs)
    (initial : State Store Metadata) where
  first : Runtime.Route (service.first c base scratch layout sl)
    (fun code m => Execution.groupValues (platform.scalar code.object.primitives) code.plan api
      (groupBuffers base scratch layout sl) c.x c.w base.dy c.r c.mu m) initial
  last : Runtime.Route (service.last c base scratch layout)
    (fun code m => Execution.finalValues (platform.scalar code.object.primitives) code.finalPlan api
      (finalBuffers base layout) (min (Geometry.groups c.columns) c.rows) scratch.dw scratch.db m) first.final

variable {service : Service platform api neg inputs Metadata} {base : Base} {scratch : Scratch}
  {layout : BaseLayout base c inputs} {sl : ScratchLayout base scratch c inputs}

def LaunchesRoute.final (r : LaunchesRoute service c base scratch layout sl initial) := r.last.final

theorem launches_route (ok : ((launches service c base scratch layout sl).run initial).result=.ok ()) :
    ∃ r : LaunchesRoute service c base scratch layout sl initial,
      ((launches service c base scratch layout sl).run initial).state=r.final := by
  obtain ⟨unit,hf,hl,hs,_⟩ := bind_success _ _ _ _ ok
  cases unit
  obtain ⟨fr,hfs⟩ := Runtime.dispatch_route hf
  rw [hfs] at hl hs
  obtain ⟨lr,hls⟩ := Runtime.dispatch_route hl
  exact ⟨⟨fr,lr⟩,hs.trans hls⟩

inductive LaunchBranch (service : Service platform api neg inputs Metadata) (c : SavedContext)
    (data : All) (ready : Ready data c inputs) (initial : State Store Metadata) where
  | skipped (none : data.scratch=Option.none)
  | ran (scratch : Scratch) (some : data.scratch=Option.some scratch)
      (route : LaunchesRoute service c data.base scratch ready.base (ready.scratch scratch some) initial)

def LaunchBranch.final {data : All} {ready : Ready data c inputs} :
    LaunchBranch service c data ready initial → State Store Metadata
  | .skipped _ => initial | .ran _ _ r => r.final

theorem branch_none (service : Service platform api neg inputs Metadata) (c : SavedContext)
    (data : All) (ready : Ready data c inputs) (hn : data.scratch=none) :
    launchPrepared service c data ready=Action.pure () := by
  unfold launchPrepared
  split
  · rfl
  · rename_i sc hs
    rw [hn] at hs
    cases hs

theorem branch_some (service : Service platform api neg inputs Metadata) (c : SavedContext)
    (data : All) (ready : Ready data c inputs) (sc : Scratch) (hsc : data.scratch=some sc) :
    launchPrepared service c data ready=
      launches service c data.base sc ready.base (ready.scratch sc hsc) := by
  unfold launchPrepared
  split
  · rename_i hs
    rw [hs] at hsc
    cases hsc
  · rename_i sc' hs
    have he : sc'=sc := Option.some.inj (hs.symm.trans hsc)
    subst sc'
    rfl

structure BodyRoute (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (service : Service platform api neg inputs Metadata) (c : SavedContext) (b : Dense) (saved : Saved c b)
    (dy : Storage.Tensor) (initial : State Store Metadata) where
  preparation : PreparationRoute lib extra c b dy initial
  ready : Ready preparation.data c inputs
  launch : LaunchBranch service c preparation.data ready preparation.final
  returns : ReturnRoute lib c preparation.data.base launch.final

variable {saved : Saved c b}
def BodyRoute.final (r : BodyRoute lib extra service c b saved dy initial) := r.returns.final
def BodyRoute.output (r : BodyRoute lib extra service c b saved dy initial) := r.returns.output

theorem body_route (out : List (Option Storage.Tensor))
    (ok : ((body lib extra service c b saved dy).run initial).result=.ok out) :
    ∃ r : BodyRoute lib extra service c b saved dy initial,r.output=out ∧
      ((body lib extra service c b saved dy).run initial).state=r.final := by
  obtain ⟨p,hp,hl,hsp,_⟩ := bind_success _ _ _ _ ok
  have hprep := certify_success _ _ _ initial p hp
  obtain ⟨pr,he,hstate⟩ := preparation_route p.val hprep
  rcases p with ⟨p,ready⟩
  dsimp only at he
  subst p
  have hpstate : ((prepared lib extra c b saved dy).run initial).state=pr.final := hstate
  rw [hpstate] at hl hsp
  obtain ⟨unit,hlaunch,hret,hsl,_⟩ := bind_success _ _ _ _ hl
  cases unit
  dsimp only at hret hsl hsp hlaunch
  unfold launchPrepared at hlaunch
  split at hlaunch
  · rename_i hn
    have hs : pr.data.scratch=none := hn
    rw [branch_none service c pr.data ready hs] at hret hsl hsp
    simp only [Action.pure] at hret hsl hsp
    obtain ⟨rr,ho,hrstate⟩ := finish_route out hret
    exact ⟨⟨pr,ready,.skipped hs,rr⟩,ho,hsp.trans (hsl.trans hrstate)⟩
  · rename_i sc hsc
    obtain ⟨lr,hlstate⟩ := launches_route hlaunch
    rw [branch_some service c pr.data ready sc hsc] at hret hsl hsp
    rw [hlstate] at hret hsl
    obtain ⟨rr,ho,hrstate⟩ := finish_route out hret
    exact ⟨⟨pr,ready,.ran sc hsc lr,rr⟩,ho,hsp.trans (hsl.trans hrstate)⟩

#axiomsClean preparation_route
#axiomsClean launches_route
#axiomsClean body_route
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
