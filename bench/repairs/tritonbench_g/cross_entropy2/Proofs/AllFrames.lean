import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelFrames

namespace VeriTile.Bench.Repairs.CrossEntropy2.Frames
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind Observations IntegerViews)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Retains bytes Word)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

variable {α σ : Type} [Inhabited α] {obs : Observations α σ} {layout : Layout σ} {kept : RegionName → Prop}
  {rt : Runtime α σ}

theorem aggregate_frame (lib : LocalLibrary obs layout kept rt) (o : Options) (count : Nat)
    (labels loss lse : Tensor) (old before : World α σ) (frame : Frame obs layout kept old before)
    (lossFresh : ¬old.live loss.region) (labelType : labels.dtype="torch.int32") :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old)
      (aggregate rt o count labels loss lse before) := by
  unfold aggregate
  apply bind_post _ _ (fun pair after => Frame obs layout kept old after ∧ ¬old.live pair.1.region) _ _
  · split
    · apply bind_post _ _ _ _ _ (fresh_frame lib old before frame _ (.lse lse))
      intro nextLSE logged loggedFacts
      apply bind_post _ _ _ _ _ (fresh_frame lib old logged loggedFacts.1 _ (.sum loss))
      intro nextLoss summed summedFacts
      exact ⟨summedFacts.1,summedFacts.2.2⟩
    · exact ⟨frame,lossFresh⟩
  · intro pair reduced reducedFacts
    apply bind_post _ _ _ _ _ (write_frame lib old reduced reducedFacts.1 pair.1 reducedFacts.2 _ (.add pair.2))
    intro addedLoss added addedFacts
    have addedFresh : ¬old.live addedLoss.region := by simpa only [addedFacts.1] using reducedFacts.2
    apply bind_post _ _ _ _ _ (fresh_frame lib old added addedFacts.2 _ (.scale o.squareScale pair.2))
    intro scaled scaledState scaleFacts
    apply bind_post _ _ _ _ _ (fresh_frame lib old scaledState scaleFacts.1 _ (.multiply scaled pair.2))
    intro zLoss multiplied multiplyFacts
    apply bind_post _ _ _ _ _ (write_frame lib old multiplied multiplyFacts.1 addedLoss addedFresh _ (.add zLoss))
    intro correctedLoss corrected correctionFacts
    have correctedFresh : ¬old.live correctedLoss.region := by simpa only [correctionFacts.1] using addedFresh
    apply bind_post _ _ _ _ _ (fresh_frame lib old corrected correctionFacts.2 _ (.widen labels labelType))
    intro wideLabels widened wideFacts
    apply bind_post _ _ _ _ _ (fresh_frame lib old widened wideFacts.1 _ (.compare wideLabels o.ignored))
    intro mask compared compareFacts
    apply bind_post _ _ _ _ _ (write_frame lib old compared compareFacts.1 correctedLoss correctedFresh _ (.mask mask))
    intro maskedLoss masked maskFacts
    apply bind_post _ _ _ _ _ (write_frame lib old masked maskFacts.2 zLoss multiplyFacts.2.2 _ (.mask mask))
    intro maskedZ final finalFacts
    exact finalFacts.2

theorem forward_frame_shaped (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (M : FP32 α) (o : Options) (x y : Tensor) (rows cols : Nat) (initial : World α σ)
    (shape : x.shape=[rows,cols]) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (forward M rt o x y initial) := by
  classical
  simp only [Host.forward,shape]
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
    apply bind_post _ _ _ _ _ (forward_launch_frame lib ints M _ _ initial allocatedZ az.1 al.2.2 ae.2.2 az.2.2)
    intro _ computed gridFrame
    split
    · exact aggregate_frame lib o _ labels loss lse initial computed gridFrame al.2.2 (ny.2.2.trans admitted.2.2.1)
    · exact gridFrame
  · exact Frame.refl _ _ _ _

theorem backward_frame_shaped (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (M : FP32 α) (o : Options) (g x l y : Tensor) (rows cols : Nat) (initial : World α σ)
    (shape : x.shape=[rows,cols]) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (backward M rt o g x l y initial) := by
  classical
  simp only [Host.backward,shape]
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
    apply bind_post _ _ _ _ _ (backward_launch_frame lib ints M _ _ initial allocated allocation.1 allocation.2.2)
    intro _ computed gridFrame
    exact gridFrame
  · exact Frame.refl _ _ _ _

/-- Every success and every local failure preserves caller storage. Invalid
rank rejection returns the unchanged initial state as well. -/
theorem forward_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (M : FP32 α) (o : Options) (x y : Tensor) (initial : World α σ) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (forward M rt o x y initial) := by
  cases shape : x.shape with
  | nil => simpa only [Host.forward,shape,Post] using Frame.refl obs layout kept initial
  | cons rows tail =>
    cases tail with
    | nil => simpa only [Host.forward,shape,Post] using Frame.refl obs layout kept initial
    | cons cols tail =>
      cases tail with
      | nil => exact forward_frame_shaped lib ints M o x y rows cols initial shape
      | cons d rest => simpa only [Host.forward,shape,Post] using Frame.refl obs layout kept initial

theorem backward_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (M : FP32 α) (o : Options) (g x l y : Tensor) (initial : World α σ) :
    Post (fun _ after => Frame obs layout kept initial after) (Frame obs layout kept initial)
      (backward M rt o g x l y initial) := by
  cases shape : x.shape with
  | nil => simpa only [Host.backward,shape,Post] using Frame.refl obs layout kept initial
  | cons rows tail =>
    cases tail with
    | nil => simpa only [Host.backward,shape,Post] using Frame.refl obs layout kept initial
    | cons cols tail =>
      cases tail with
      | nil => exact backward_frame_shaped lib ints M o g x l y rows cols initial shape
      | cons d rest => simpa only [Host.backward,shape,Post] using Frame.refl obs layout kept initial

theorem Frame.all_bytes (encode : α → Word) {before after : World α σ} (frame : Frame obs layout kept before after)
    (r : RegionName) (retained : kept r) (live : before.live r) (active : layout.active before.native r)
    (offset : Nat) (inside : offset<(layout.buffer before.native r).capacity) :
    bytes layout encode after (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      bytes layout encode before (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset) := by
  exact VeriTile.Bench.Repairs.CrossEntropy1.Physical.preserved_allocation_bytes layout encode obs before after r frame.storage live active
    (frame.lifetime.active r retained active) (frame.lifetime.buffer r retained active).symm offset inside

#axiomsClean aggregate_frame
#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean Frame.all_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Frames
