import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Invocation
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AllFrames

namespace VeriTile.Bench.Repairs.CrossEntropy2.Invocation
open VeriTile.Triton Source Host Floating Frames
open Floating.Profiled (BackendABI BackwardABI DeviceFamily)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations IntegerViews)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000
variable {σ : Type} {obs : Observations Compiled σ} {layout : Layout σ} {kept : RegionName → Prop}
  {rt : Runtime Compiled σ}

theorem mark_frame {A B : Type} (tag : A) (call : Step (World Compiled σ) B) (old : World Compiled σ)
    (frame : Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old) call) :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old) (mark tag call) := by
  cases call <;> exact frame

theorem aggregate_frame (lib : LocalLibrary obs layout kept rt) (abi : BackendABI) (o : Options) (count : Nat)
    (labels loss lse : Tensor) (old before : World Compiled σ) (frame : Frame obs layout kept old before)
    (lossFresh : ¬old.live loss.region) (labelType : labels.dtype="torch.int32") :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old)
      (aggregate layout abi rt o count labels loss lse before) := by
  unfold aggregate
  apply bind_post _ _ (fun chosen after => Frame obs layout kept old after ∧ ¬old.live chosen.2.1.region) _ _
  · split
    · apply bind_post _ _ _ _ _ (fresh_frame lib old before frame _ (.lse lse))
      intro nextLSE logged loggedFacts
      apply bind_post _ _ _ _ _ (fresh_frame lib old logged loggedFacts.1 _ (.sum loss))
      intro nextLoss summed summedFacts
      exact ⟨summedFacts.1,summedFacts.2.2⟩
    · exact ⟨frame,lossFresh⟩
  · intro chosen reduced facts
    apply mark_frame
    have original := Frames.aggregate_frame lib o 1 labels chosen.2.1 chosen.2.2 old reduced facts.1 facts.2 labelType
    simpa only [Host.aggregate,show ¬(1:Nat)>1 by decide,↓reduceIte,
      VeriTile.Bench.Repairs.CrossEntropy1.Host.bind,finish] using original

theorem forward_launch_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (abi : BackendABI) (p : Parameters) (count : Nat) (old before : World Compiled σ)
    (frame : Frame obs layout kept old before)
    (lossFresh : ¬old.live p.loss) (lseFresh : ¬old.live p.lse) (zFresh : ¬old.live p.zLoss) :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old)
      (forwardLaunch layout abi rt p count before) := by
  unfold forwardLaunch
  split
  · exact frame
  · apply bind_post _ _ _ _ _ (setup_frame lib old before frame _ (.forward p count))
    intro _ ready readyFrame
    let M := Floating.Profiled.algebra (abi.device (Floating.Scheduled.keyFor layout ready p)) p.logitScale
    apply grid_frame ints old ready readyFrame (fun m => Grid.forward M p m count)
    intro r live i
    exact forward_other M p ready.memory count r i (fun eq => lossFresh (eq ▸ live))
      (fun eq => lseFresh (eq ▸ live)) (fun eq => zFresh (eq ▸ live))

theorem backward_launch_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (abi : BackwardABI) (device : DeviceFamily) (p : Parameters) (count : Nat) (old before : World Compiled σ)
    (frame : Frame obs layout kept old before) (fresh : ¬old.live p.gradient) :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old)
      (backwardLaunch layout abi device rt p count before) := by
  unfold backwardLaunch
  split
  · exact frame
  · apply bind_post _ _ _ _ _ (setup_frame lib old before frame _ (.backward p count))
    intro _ ready readyFrame
    let M := Floating.Profiled.algebra device p.logitScale
    apply grid_frame ints old ready readyFrame (fun m => backwardGrid M p m count)
    intro r live i
    exact backward_other M p ready.memory count r i (fun eq => fresh (eq ▸ live))

theorem forward_frame_shaped (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (abi : BackendABI) (o : Options) (x y : Tensor) (rows cols : Nat) (initial : World Compiled σ)
    (shape : x.shape=[rows,cols]) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (forward layout abi rt o x y initial) := by
  classical
  simp only [forward,shape]
  split
  · rename_i admitted
    apply bind_post _ _ _ _ _ (normalize_frame lib initial initial (Frame.refl _ _ _ _) x)
    intro logits normalizedX nx
    apply bind_post _ _ _ _ _ (normalize_frame lib initial normalizedX nx.1 y)
    intro labels normalizedY ny
    apply bind_post _ _ _ _ _ (fresh_frame lib initial normalizedY ny.1 _ (.empty _ _ _))
    intro loss allocatedLoss al
    apply bind_post _ _ _ _ _ (fresh_frame lib initial allocatedLoss al.1 _ (.empty _ _ _))
    intro lse allocatedLSE ae
    apply bind_post _ _ _ _ _ (fresh_frame lib initial allocatedLSE ae.1 _ (.empty _ _ _))
    intro zLoss allocatedZ az
    apply bind_post _ _ _ _ _ (forward_launch_frame lib ints abi _ _ initial allocatedZ az.1 al.2.2 ae.2.2 az.2.2)
    intro device computed gridFrame
    split
    · apply mark_frame
      exact aggregate_frame lib abi o _ labels loss lse initial computed gridFrame al.2.2 (ny.2.2.trans admitted.2.2.1)
    · exact gridFrame
  · exact Frame.refl _ _ _ _

theorem backward_frame_shaped (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (abi : BackwardABI) (device : DeviceFamily) (o : Options) (g x l y : Tensor)
    (rows cols : Nat) (initial : World Compiled σ) (shape : x.shape=[rows,cols]) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (backward layout abi device rt o g x l y initial) := by
  classical
  simp only [backward,shape]
  split
  · apply bind_post _ _ _ _ _ (normalize_frame lib initial initial (Frame.refl _ _ _ _) x)
    intro logits normalizedX nx
    apply bind_post _ _ _ _ _ (normalize_frame lib initial normalizedX nx.1 y)
    intro labels normalizedY ny
    apply bind_post _ _ _ _ _ (normalize_frame lib initial normalizedY ny.1 g)
    intro upstream normalizedG ng
    apply bind_post _ _ _ _ _ (normalize_frame lib initial normalizedG ng.1 l)
    intro lse normalizedL nl
    apply bind_post _ _ _ _ _ (fresh_frame lib initial normalizedL nl.1 _ (.emptyLike logits Bool.true))
    intro gradient allocated allocation
    apply bind_post _ _ _ _ _ (backward_launch_frame lib ints abi device _ _ initial allocated allocation.1 allocation.2.2)
    intro _ computed gridFrame
    exact gridFrame
  · exact Frame.refl _ _ _ _

theorem forward_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (abi : BackendABI) (o : Options) (x y : Tensor) (initial : World Compiled σ) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (forward layout abi rt o x y initial) := by
  cases shape : x.shape with
  | nil => simpa only [forward,shape,Post] using Frame.refl obs layout kept initial
  | cons rows tail =>
    cases tail with
    | nil => simpa only [forward,shape,Post] using Frame.refl obs layout kept initial
    | cons cols tail =>
      cases tail with
      | nil => exact forward_frame_shaped lib ints abi o x y rows cols initial shape
      | cons d rest => simpa only [forward,shape,Post] using Frame.refl obs layout kept initial

theorem backward_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (abi : BackwardABI) (device : DeviceFamily) (o : Options) (g x l y : Tensor) (initial : World Compiled σ) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (backward layout abi device rt o g x l y initial) := by
  cases shape : x.shape with
  | nil => simpa only [backward,shape,Post] using Frame.refl obs layout kept initial
  | cons rows tail =>
    cases tail with
    | nil => simpa only [backward,shape,Post] using Frame.refl obs layout kept initial
    | cons cols tail =>
      cases tail with
      | nil => exact backward_frame_shaped lib ints abi device o g x l y rows cols initial shape
      | cons d rest => simpa only [backward,shape,Post] using Frame.refl obs layout kept initial

#axiomsClean forward_frame
#axiomsClean backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Invocation
