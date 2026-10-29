import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostAPIReal
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostPropagation

namespace VeriTile.Bench.Repairs.ParallelAttention.HostAPI
noncomputable section
open scoped BigOperators

/-- The O/Z handles retained by the ordinary division graph still contain the
actual forward values, including when the public result uses a new buffer. -/
theorem forward_raw (p : Plans) (s : Shape) (domain : ShapeContract s)
    (x : Inputs) (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (layout : RunLayout s (forwardBases x) trees scratch output forwardKinds ks hs)
    (entry : Memory ℝ) (eps : ℝ) (normalized : Nat) (returnBoth normalize : Bool)
    (fresh : normalize=true → returnBoth=false → NormalizationLayout s output normalized)
    (bh : Nat) (hb : bh<s.B*s.H) :
    let result := forward realOps p s x trees scratch output ks hs entry eps normalized returnBoth normalize
    (∀ t,t<s.T → ∀ e,e<s.V → result.rawOutput.read result.memory bh t e=
      outputSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t e) ∧
    (∀ t,t<s.T → result.rawNormalizer.read result.memory bh t 0=
      normalizerSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t) := by
  have frame (address : Nat)
      (member : address∈normalizeReads (s.B*s.H) s.T s.V (output .forward) (output .normalizer)) :
      forwardMemory realOps p s x trees scratch output ks hs entry eps normalized returnBoth normalize address=
        combinedRun realOps p s (forwardBases x) trees scratch output ks hs entry address := by
    unfold forwardMemory
    rw [dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout entry]
    split_ifs with both norm
    · rfl
    · apply runStores_frame
      intro i hi eq
      exact fresh norm (by cases returnBoth <;> simp_all) i (List.mem_range.mp hi) (eq ▸ member)
    · rfl
  have correct := (sourceCorrect p s domain (forwardBases x) trees scratch output forwardKinds ks hs layout entry).1
  constructor
  · intro t ht e he
    change forwardMemory _ _ _ _ _ _ _ _ _ _ _ _ _ _ (output .forward+dense s.T s.V bh t e)=_
    rw [frame _ (Or.inl ⟨Nat.le_add_right _ _,Nat.add_lt_add_left (dense_bound _ _ _ _ _ _ hb ht he) _⟩)]
    exact correct .forward (by simp [forwardKinds]) bh t e hb ht he
  · intro t ht
    change forwardMemory _ _ _ _ _ _ _ _ _ _ _ _ _ _ (output .normalizer+dense s.T 1 bh t 0)=_
    rw [frame _ (Or.inr ⟨Nat.le_add_right _ _,by
      simpa using Nat.add_lt_add_left (dense_bound s.T 1 (s.B*s.H) bh t 0 hb ht (by omega)) (output .normalizer)⟩)]
    exact correct .normalizer (by simp [forwardKinds]) bh t 0 hb ht (by norm_num [OutputKind.width])

/-- The later ordinary division backward reads these retained Tensor values.
This is a saved-tensor/version premise, not an O/Z numerical-answer premise. -/
def propagated (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (later : Memory ℝ) (bh : Nat) : Cotangent ℝ :=
  if normalize then
    ⟨fun t e => if bh<s.B*s.H ∧ t<s.T ∧ e<s.V then
        viewRead realOps s s.V dy later bh t e / (z.read later bh t 0+eps) else 0,
      fun t => if bh<s.B*s.H ∧ t<s.T then -(∑ e∈Finset.range s.V,
        viewRead realOps s s.V dy later bh t e*o.read later bh t e)/(z.read later bh t 0+eps)^2 else 0⟩
  else outputOnlyCotangent (viewRead realOps s s.V dy later bh)

def cotangentValue (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (later : Memory ℝ) (event : Event) : ℝ :=
  let D := event.1.width s
  let bh := event.2/D/s.T%(s.B*s.H)
  let t := event.2/D%s.T
  let e := event.2%D
  match event.1 with
  | .forward => (propagated s eps normalize o z dy later bh).output t e
  | .normalizer => (propagated s eps normalize o z dy later bh).normalizer t
  | _ => 0

/-- Only active elements of the retained O/Z and incoming dy tensors are
read by the current-memory host cotangent graph. -/
theorem propagated_local (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (m m' : Memory ℝ)
    (same : Set.EqOn m m' (propagationReads s o z dy)) (bh : Nat) :
    propagated s eps normalize o z dy m bh=propagated s eps normalize o z dy m' bh := by
  have hdy : viewRead realOps s s.V dy m bh=viewRead realOps s s.V dy m' bh := by
    funext t e
    unfold viewRead
    split_ifs with active
    · exact same (Or.inr ⟨bh,t,e,active.1,active.2.1,active.2.2,rfl⟩)
    · rfl
  have ho (t e : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.V) :
      o.read m bh t e=o.read m' bh t e :=
    same (Or.inl (Or.inl ⟨bh,t,e,hb,ht,he,rfl⟩))
  have hz (t : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) : z.read m bh t 0=z.read m' bh t 0 :=
    same (Or.inl (Or.inr ⟨bh,t,hb,ht,rfl⟩))
  cases normalize with
  | false => simp only [propagated,Bool.false_eq_true,if_false,hdy]
  | true =>
    unfold propagated
    simp only [if_true,hdy]
    congr 1
    · funext t e
      split_ifs with active
      · rw [hz t active.1 active.2.1]
      · rfl
    · funext t
      split_ifs with active
      · rw [hz t active.1 active.2]
        congr 2
        apply Finset.sum_congr rfl
        intro e he
        rw [ho t e active.1 active.2 (Finset.mem_range.mp he)]
      · rfl

def propagateRun (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (later : Memory ℝ) (dest : OutputKind → Nat) : Memory ℝ :=
  runTagged (tensorAddress dest) (fun current ev => cotangentValue s eps normalize o z dy current ev)
    (events s false forwardKinds) later

theorem propagate_output (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (later : Memory ℝ) (dest : OutputKind → Nat)
    (layout : PropagationLayout s dest) (fresh : PropagationFresh s o z dy dest) (bh t e : Nat)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.V) :
    propagateRun s eps normalize o z dy later dest (dest .forward+dense s.T s.V bh t e)=
      (propagated s eps normalize o z dy later bh).output t e := by
  have h := tagged_output (propagationReads s o z dy) (tensorAddress dest)
    (fun current ev => cotangentValue s eps normalize o z dy current ev)
    (by
      intro m m' same ev
      unfold cotangentValue
      simp_rw [propagated_local s eps normalize o z dy m m' same])
    (events s false forwardKinds) later fresh layout
    (.forward,dense s.T s.V bh t e)
    ((events_member _ _ _ _ _).mpr ⟨by simp [forwardKinds],dense_bound _ _ _ _ _ _ hb ht he⟩)
  have hd := scratch_decode (s.B*s.H) s.T s.V 0 bh t e hb ht he
  simp only [zero_mul,zero_add] at hd
  simpa [propagateRun,tensorAddress,cotangentValue,OutputKind.width,hd.1,hd.2.1,hd.2.2.1] using h

theorem propagate_normalizer (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (later : Memory ℝ) (dest : OutputKind → Nat)
    (layout : PropagationLayout s dest) (fresh : PropagationFresh s o z dy dest) (bh t : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) :
    propagateRun s eps normalize o z dy later dest (dest .normalizer+dense s.T 1 bh t 0)=
      (propagated s eps normalize o z dy later bh).normalizer t := by
  have h := tagged_output (propagationReads s o z dy) (tensorAddress dest)
    (fun current ev => cotangentValue s eps normalize o z dy current ev)
    (by
      intro m m' same ev
      unfold cotangentValue
      simp_rw [propagated_local s eps normalize o z dy m m' same])
    (events s false forwardKinds) later fresh layout
    (.normalizer,dense s.T 1 bh t 0)
    ((events_member _ _ _ _ _).mpr ⟨by simp [forwardKinds],dense_bound _ _ _ _ _ _ hb ht (by norm_num)⟩)
  have hd := scratch_decode (s.B*s.H) s.T 1 0 bh t 0 hb ht (by norm_num)
  simp only [zero_mul,zero_add] at hd
  simpa only [propagateRun,tensorAddress,cotangentValue,OutputKind.width,hd.1,hd.2.1,hd.2.2.1] using h

theorem propagate_frame (s : Shape) (eps : ℝ) (normalize : Bool)
    (o z : Tensor) (dy : View) (later : Memory ℝ) (dest : OutputKind → Nat) (address : Nat)
    (outside : ∀ e∈events s false forwardKinds,tensorAddress dest e≠address) :
    propagateRun s eps normalize o z dy later dest address=later address :=
  tagged_frame _ _ _ _ _ outside

/-- The explicit host arithmetic agrees with the independent normalization
cotangent on active elements, derived from already proved forward memory. -/
theorem propagated_correct (p : Plans) (s : Shape) (domain : ShapeContract s)
    (x : Inputs) (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (layout : RunLayout s (forwardBases x) trees scratch output forwardKinds ks hs)
    (entry later : Memory ℝ) (eps : ℝ) (normalized : Nat) (normalize : Bool)
    (fresh : normalize=true → NormalizationLayout s output normalized) (dy : View)
    (preserved : let f := forward realOps p s x trees scratch output ks hs entry eps normalized false normalize
      RawSaved s f.rawOutput f.rawNormalizer f.memory later)
    (bh : Nat) (hb : bh<s.B*s.H) :
    let f := forward realOps p s x trees scratch output ks hs entry eps normalized false normalize
    let g := propagated s eps normalize f.rawOutput f.rawNormalizer dy later bh
    let target := if normalize then normalizedCotangent s.T s.K s.V (selectedScale s.K s.useScale) eps
      (inputValues realOps s x entry bh) (viewRead realOps s s.V dy later bh)
      else outputOnlyCotangent (viewRead realOps s s.V dy later bh)
    (∀ t,t<s.T → ∀ e,e<s.V → g.output t e=target.output t e) ∧
    (∀ t,t<s.T → g.normalizer t=target.normalizer t) := by
  have raw := forward_raw p s domain x trees scratch output ks hs layout entry eps normalized false normalize
    (by intro h _;exact fresh h) bh hb
  cases normalize with
  | false => exact ⟨by intros;rfl,by intros;rfl⟩
  | true =>
    constructor
    · intro t ht e he
      simp only [propagated,if_true,hb,ht,he,and_self,if_pos]
      change _ / (_+eps)=_ / (_+eps)
      rw [preserved.2 bh t hb ht,raw.2 t ht]
    · intro t ht
      simp only [propagated,if_true,hb,ht,and_self,if_pos]
      change -(∑ e∈Finset.range s.V,_*_)/(_+eps)^2=-(∑ e∈Finset.range s.V,_*_)/(_+eps)^2
      rw [preserved.2 bh t hb ht,raw.2 t ht]
      congr 2
      apply Finset.sum_congr rfl
      intro e he
      rw [preserved.1 bh t e hb ht (Finset.mem_range.mp he),raw.1 t ht e (Finset.mem_range.mp he)]

/-- Loss equality needs only active cotangent elements; masks outside the
public tensor are deliberately not equated to out-of-bounds expressions. -/
theorem loss_cotangent_congr (T K V : Nat) (scale : ℝ) (x : ParallelAttention.Inputs ℝ)
    (g g' : Cotangent ℝ)
    (ho : ∀ t,t<T → ∀ e,e<V → g.output t e=g'.output t e)
    (hz : ∀ t,t<T → g.normalizer t=g'.normalizer t) :
    lossSpec T K V scale x g=lossSpec T K V scale x g' := by
  unfold lossSpec
  apply Finset.sum_congr rfl
  intro t ht
  rw [hz t (Finset.mem_range.mp ht)]
  congr 1
  apply Finset.sum_congr rfl
  intro e he
  rw [ho t (Finset.mem_range.mp ht) e (Finset.mem_range.mp he)]

def publicSingleLoss (s : Shape) (eps : ℝ) (normalize : Bool)
    (x : ParallelAttention.Inputs ℝ) (dy : Nat → Nat → ℝ) : ℝ :=
  if normalize then normalizedLoss s.T s.K s.V (selectedScale s.K s.useScale) eps x dy
  else lossSpec s.T s.K s.V (selectedScale s.K s.useScale) x (outputOnlyCotangent dy)

/-- Separate public backward after real host cotangent propagation. Its only
later-state assumptions preserve saved inputs/O/Z and allocate disjoint buffers.
No premise states the normalized cotangent or any desired gradient value. -/
theorem single_backward_vjp (p : Plans) (s : Shape) (domain : ShapeContract s) (x : Inputs) (_metadata : x.Valid s)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (fLayout : RunLayout s (forwardBases x) fTrees fScratch fOutput forwardKinds fks fhs)
    (entry later : Memory ℝ) (eps : ℝ) (normalized : Nat) (normalize : Bool)
    (fresh : normalize=true → NormalizationLayout s fOutput normalized) (dy : View)
    (preserved : let f := forward realOps p s x fTrees fScratch fOutput fks fhs entry eps normalized false normalize
      RawSaved s f.rawOutput f.rawNormalizer f.memory later)
    (dest : OutputKind → Nat) (propLayout : PropagationLayout s dest)
    (propReadsFresh : let f := forward realOps p s x fTrees fScratch fOutput fks fhs entry eps normalized false normalize
      PropagationFresh s f.rawOutput f.rawNormalizer dy dest)
    (saved : SavedValid (save realOps s x) entry later)
    (propFresh : ∀ e∈events s false forwardKinds,tensorAddress dest e∉inputCellsOnly s x)
    (bTrees : OutputKind → SumPlan) (bScratch bOutput : OutputKind → Nat) (bks bhs : List Event)
    (bLayout : RunLayout s (backwardBases (save realOps s x)
      (tensor s .forward (dest .forward)) (tensor s .normalizer (dest .normalizer)))
      bTrees bScratch bOutput backwardKinds bks bhs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ)
    (nonzero : normalize=true → ∀ t,t<s.T →
      normalizerSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t+eps≠0) :
    let f := forward realOps p s x fTrees fScratch fOutput fks fhs entry eps normalized false normalize
    let ready := propagateRun s eps normalize f.rawOutput f.rawNormalizer dy later dest
    let result := backward realOps p f.saved (tensor s .forward (dest .forward))
      (tensor s .normalizer (dest .normalizer)) bTrees bScratch bOutput bks bhs ready
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => publicSingleLoss s eps normalize
      (affineInputs (inputValues realOps s x entry bh) dx a) (viewRead realOps s s.V dy later bh))
      (result.returned.pairing s bh dx result.memory) 0 := by
  let f := forward realOps p s x fTrees fScratch fOutput fks fhs entry eps normalized false normalize
  let ready := propagateRun s eps normalize f.rawOutput f.rawNormalizer dy later dest
  let do_ := tensor s .forward (dest .forward)
  let dz := tensor s .normalizer (dest .normalizer)
  have valid : SavedValid (save realOps s x) entry ready := by
    intro address ha
    calc
      ready address=later address := by
        apply propagate_frame
        intro e he eq
        exact propFresh e he (eq ▸ ha)
      _=entry address := saved ha
  have hx := saved_inputs realOps (save realOps s x) entry ready valid do_ dz bh
  change headInputs (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh=
    inputValues realOps s x entry bh at hx
  have computed := propagated_correct p s domain x fTrees fScratch fOutput fks fhs fLayout
    entry later eps normalized normalize fresh dy preserved bh hb
  let target := if normalize then normalizedCotangent s.T s.K s.V (selectedScale s.K s.useScale) eps
      (inputValues realOps s x entry bh) (viewRead realOps s s.V dy later bh)
      else outputOnlyCotangent (viewRead realOps s s.V dy later bh)
  have ho (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) :
      (headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh).output t e=
        target.output t e := by
    change tensorRead realOps ready (dest .forward) (s.B*s.H) s.T s.V bh t e=_
    rw [tensorRead,if_pos ⟨hb,ht,he⟩]
    exact (propagate_output s eps normalize f.rawOutput f.rawNormalizer dy later dest propLayout propReadsFresh bh t e hb ht he).trans
      (computed.1 t ht e he)
  have hz (t : Nat) (ht : t<s.T) :
      (headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh).normalizer t=
        target.normalizer t := by
    change tensorRead realOps ready (dest .normalizer) (s.B*s.H) s.T 1 bh t 0=_
    rw [tensorRead,if_pos ⟨hb,ht,by norm_num⟩]
    exact (propagate_normalizer s eps normalize f.rawOutput f.rawNormalizer dy later dest propLayout propReadsFresh bh t hb ht).trans
      (computed.2 t ht)
  have h := returned_vjp p s domain (backwardBases (save realOps s x) do_ dz)
    bTrees bScratch bOutput bks bhs bLayout ready bh hb dx
  rw [hx] at h
  have lossEq (a : ℝ) := loss_cotangent_congr s.T s.K s.V (selectedScale s.K s.useScale)
    (affineInputs (inputValues realOps s x entry bh) dx a) _ target ho hz
  simp_rw [lossEq] at h
  have dispatchEq := dispatch_eq realOps p s (backwardBases (save realOps s x) do_ dz)
    bTrees bScratch bOutput backwardKinds bks bhs bLayout ready
  constructor
  · rfl
  · simp only [forward,backward,saved_ops_eq]
    change HasDerivAt (fun a : ℝ => publicSingleLoss s eps normalize
        (affineInputs (inputValues realOps s x entry bh) dx a) (viewRead realOps s s.V dy later bh))
      (returnedPairing s bh dx bOutput
        (dispatch realOps p s (backwardBases (save realOps s x) do_ dz) bTrees bScratch bOutput bks bhs ready)) 0
    rw [dispatchEq]
    cases normalize with
    | false => exact h
    | true =>
      have same := h.unique (independent_vjp s.T s.K s.V (selectedScale s.K s.useScale)
        (inputValues realOps s x entry bh) dx target)
      rw [same]
      exact normalized_vjp s.T s.K s.V (selectedScale s.K s.useScale) eps
        (inputValues realOps s x entry bh) dx (viewRead realOps s s.V dy later bh) (nonzero rfl)


end
end VeriTile.Bench.Repairs.ParallelAttention.HostAPI
