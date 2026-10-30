import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostNative
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostNormalize
import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeExecutionReal

namespace VeriTile.Bench.Repairs.ParallelAttention.HostNative
noncomputable section
open scoped BigOperators
open HostAPI

/-- The actual kernel path, including the wrapper's explicit empty dispatch. -/
theorem dispatch_real (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory ℝ) :
    dispatch realOps c s b trees scratch output ks hs m=
      HostAPI.dispatch realOps c.plans s b trees scratch output ks hs m := by
  simp only [dispatch,HostAPI.dispatch,compiledRun_real]

theorem forward_correct (c : CompilerLayout) (s : Shape) (domain : ShapeContract s)
    (x : HostAPI.Inputs) (_metadata : x.Valid s)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s (forwardBases x) trees scratch output forwardKinds ks hs)
    (entry : Memory ℝ) (eps : ℝ) (den dst : Nat) (returnBoth normalize : Bool)
    (fresh : normalize=true → returnBoth=false → NormalizeLayout s output den dst)
    (bh : Nat) (hb : bh<s.B*s.H) :
    let f := forward realOps c s x trees scratch output ks hs entry eps den dst returnBoth normalize
    PublicResult.Agrees s.T s.V (f.api.returned.read f.api.memory bh)
      (publicSpec s.T s.K (selectedScale s.K s.useScale) eps (inputValues realOps s x entry bh) returnBoth normalize) := by
  have correct := (sourceCorrect c.plans s domain (forwardBases x) trees scratch output forwardKinds ks hs layout entry).1
  have ho (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) :=
    correct .forward (by simp [forwardKinds]) bh t e hb ht he
  have hz (t : Nat) (ht : t<s.T) :=
    correct .normalizer (by simp [forwardKinds]) bh t 0 hb ht (by norm_num [OutputKind.width])
  simp only [kernelRun,OutputKind.width,mathematicalResult,logicalSpec] at ho hz
  have draw := HostAPI.dispatch_eq realOps c.plans s (forwardBases x) trees scratch output forwardKinds ks hs layout entry
  cases returnBoth <;> cases normalize <;>
    simp only [forward,forwardMemory,returned,Returned.read,publicSpec,PublicResult.Agrees,
      Bool.false_eq_true,if_false,if_true,dispatch_real,draw]
  · intro t ht e he
    exact ho t ht e he
  · intro t ht e he
    let raw := combinedRun realOps c.plans s (forwardBases x) trees scratch output ks hs entry
    change divisionRun realOps s (output .forward) den dst
      (denominatorRun realOps eps s (output .normalizer) den raw) (dst+dense s.T s.V bh t e)=_
    rw [division_output _ _ _ _ _ _ (by intro i hi;exact fun h => (fresh rfl rfl).outputFresh i hi (Or.inl h))
      _ (dense_bound _ _ _ _ _ _ hb ht he)]
    rw [denominator_frame _ _ _ _ _ _ _ (by
      intro i hi eq
      exact (fresh rfl rfl).denominatorFresh i hi (eq ▸ Or.inl
        ⟨Nat.le_add_right _ _,Nat.add_lt_add_left (dense_bound _ _ _ _ _ _ hb ht he) _⟩))]
    rw [dense_normalizer_index _ _ _ _ _ he,denominator_output _ _ _ _ _ _
      (by intro i hi member;exact (fresh rfl rfl).denominatorFresh i hi (Or.inr member))
      _ (by simpa using dense_bound s.T 1 (s.B*s.H) bh t 0 hb ht (by omega))]
    change raw (output .forward+dense s.T s.V bh t e)/(raw (output .normalizer+dense s.T 1 bh t 0)+eps)=_
    dsimp only [raw]
    rw [ho t ht e he,hz t ht]
    rfl
  · exact ⟨ho,hz⟩
  · exact ⟨ho,hz⟩

theorem backward_real (c : CompilerLayout) (s : Shape) (x : HostAPI.Inputs) (do_ dz : Tensor)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (ready : Memory ℝ) :
    backward realOps c (save realOps s x) do_ dz trees scratch output ks hs ready=
      HostAPI.backward realOps c.plans (save realOps s x) do_ dz trees scratch output ks hs ready := by
  simp only [backward,HostAPI.backward,saved_ops_eq,dispatch_real]

theorem returned_context_vjp (c : CompilerLayout) (s : Shape) (domain : ShapeContract s)
    (x : HostAPI.Inputs) (metadata : x.Valid s)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (entry : Memory ℝ) (eps : ℝ) (den dst : Nat) (returnBoth normalize : Bool)
    (incoming ready : Memory ℝ) (valid : SavedValid (save realOps s x) entry ready)
    (doView dzView : View) (do_ dz : Tensor)
    (ho : Materialized s .forward incoming ready doView do_)
    (hz : Materialized s .normalizer incoming ready dzView dz)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s (backwardBases (save realOps s x) do_ dz) trees scratch output backwardKinds ks hs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ) :
    let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst returnBoth normalize
    let result := backward realOps c f.api.saved do_ dz trees scratch output ks hs ready
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V f.api.saved.scale
      (affineInputs (inputValues realOps s x entry bh) dx a) (incomingCotangent realOps s incoming doView dzView bh))
      (result.returned.pairing s bh dx result.memory) 0 := by
  simp only [forward,backward_real]
  exact HostAPI.backward_vjp c.plans s domain x metadata entry incoming ready valid doView dzView do_ dz ho hz
    trees scratch output ks hs layout bh hb dx

/-- The selected eager Torch host division DAG. Broadcast reduction is exact
finite addition syntax bound separately to the selected host lowering. -/
theorem forward_retained (c : CompilerLayout) (s : Shape) (domain : ShapeContract s)
    (x : HostAPI.Inputs) (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (layout : RunLayout s (forwardBases x) trees scratch output forwardKinds ks hs)
    (entry : Memory ℝ) (eps : ℝ) (den dst : Nat) (fresh : NormalizeLayout s output den dst)
    (bh : Nat) (hb : bh<s.B*s.H) :
    let f := forward realOps c s x trees scratch output ks hs entry eps den dst false true
    f.denominator=some (tensor s .normalizer den) ∧
    (∀ t,t<s.T → ∀ e,e<s.V → f.api.rawOutput.read f.api.memory bh t e=
      outputSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t e) ∧
    (∀ t,t<s.T → (tensor s .normalizer den).read f.api.memory bh t 0=
      normalizerSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t+eps) := by
  let raw := dispatch realOps c s (forwardBases x) trees scratch output ks hs entry
  have draw : raw=combinedRun realOps c.plans s (forwardBases x) trees scratch output ks hs entry := by
    dsimp only [raw]
    rw [dispatch_real,HostAPI.dispatch_eq _ _ _ _ _ _ _ forwardKinds _ _ layout]
  have correct := (sourceCorrect c.plans s domain (forwardBases x) trees scratch output forwardKinds ks hs layout entry).1
  refine ⟨rfl,?_,?_⟩
  · intro t ht e he
    change divisionRun realOps s (output .forward) den dst
      (denominatorRun realOps eps s (output .normalizer) den raw) (output .forward+dense s.T s.V bh t e)=_
    have member : output .forward+dense s.T s.V bh t e∈normalizeReads (s.B*s.H) s.T s.V (output .forward) den :=
      Or.inl ⟨Nat.le_add_right _ _,Nat.add_lt_add_left (dense_bound _ _ _ _ _ _ hb ht he) _⟩
    rw [division_frame _ _ _ _ _ _ _ (by intro i hi eq;exact fresh.outputFresh i hi (Or.inl (eq ▸ member)))]
    rw [denominator_frame _ _ _ _ _ _ _ (by
      intro i hi eq
      exact fresh.denominatorFresh i hi (eq ▸ Or.inl
        ⟨Nat.le_add_right _ _,Nat.add_lt_add_left (dense_bound _ _ _ _ _ _ hb ht he) _⟩))]
    rw [draw]
    exact correct .forward (by simp [forwardKinds]) bh t e hb ht he
  · intro t ht
    change divisionRun realOps s (output .forward) den dst
      (denominatorRun realOps eps s (output .normalizer) den raw) (den+dense s.T 1 bh t 0)=_
    have hi : dense s.T 1 bh t 0<s.B*s.H*s.T := by
      simpa using dense_bound s.T 1 (s.B*s.H) bh t 0 hb ht (by omega)
    have member : den+dense s.T 1 bh t 0∈normalizeReads (s.B*s.H) s.T s.V (output .forward) den :=
      Or.inr ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hi _⟩
    rw [division_frame _ _ _ _ _ _ _ (by intro i hi eq;exact fresh.outputFresh i hi (Or.inl (eq ▸ member)))]
    rw [denominator_output _ _ _ _ _ _ (by intro i hi member;exact fresh.denominatorFresh i hi (Or.inr member)) _ hi,draw]
    have hz := correct .normalizer (by simp [forwardKinds]) bh t 0 hb ht (by norm_num [OutputKind.width])
    change combinedRun realOps c.plans s (forwardBases x) trees scratch output ks hs entry
      (output .normalizer+dense s.T 1 bh t 0)+eps=_
    simpa only [kernelRun,OutputKind.width,mathematicalResult,logicalSpec] using congrArg (fun v : ℝ => v+eps) hz

theorem propagated_real (s : Shape) (tree : Nat → Nat → SumPlan)
    (complete : ∀ bh t,(tree bh t).Covers s.V) (o den : Tensor) (dy : View)
    (m : Memory ℝ) (bh t : Nat) (hb : bh<s.B*s.H) (ht : t<s.T) :
    (propagated realOps s true tree o den dy m bh).normalizer t=
      -(∑ e∈Finset.range s.V,viewRead realOps s s.V dy m bh t e*o.read m bh t e)/(den.read m bh t 0)^2 := by
  simp only [propagated,if_true,hb,ht,and_self]
  rw [(tree bh t).real_sum s.V (complete bh t)]
  simp only [Finset.sum_div,← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro e he
  simp only [Finset.mem_range] at he
  simp only [he,if_true,realOps]
  rw [div_div]
  ring

/-- The later public call uses the retained denominator handle from forward.
Current-memory propagation preserves every saved input through fresh do/dz
allocation and uses the installed division-backward scalar operation order. -/
theorem normalized_backward_vjp (c : CompilerLayout) (s : Shape) (domain : ShapeContract s)
    (x : HostAPI.Inputs) (_metadata : x.Valid s)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (fLayout : RunLayout s (forwardBases x) fTrees fScratch fOutput forwardKinds fks fhs)
    (entry later : Memory ℝ) (eps : ℝ) (den dst : Nat) (fresh : NormalizeLayout s fOutput den dst)
    (dy : View) (tree : Nat → Nat → SumPlan) (complete : ∀ bh t,(tree bh t).Covers s.V)
    (preserved : let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false true
      RawSaved s f.api.rawOutput (tensor s .normalizer den) f.api.memory later)
    (dest : OutputKind → Nat) (propLayout : PropagationLayout s dest)
    (propFresh : PropagationFresh s (tensor s .forward (fOutput .forward)) (tensor s .normalizer den) dy dest)
    (saved : SavedValid (save realOps s x) entry later)
    (inputFresh : ∀ ev∈events s false forwardKinds,tensorAddress dest ev∉inputCellsOnly s x)
    (bTrees : OutputKind → SumPlan) (bScratch bOutput : OutputKind → Nat) (bks bhs : List Event)
    (bLayout : RunLayout s (backwardBases (save realOps s x)
      (tensor s .forward (dest .forward)) (tensor s .normalizer (dest .normalizer)))
      bTrees bScratch bOutput backwardKinds bks bhs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ)
    (nonzero : ∀ t,t<s.T → normalizerSpec s.T s.K (selectedScale s.K s.useScale)
      (inputValues realOps s x entry bh) t+eps≠0) :
    let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false true
    let ready := propagateRun realOps s true tree f.api.rawOutput (tensor s .normalizer den) dy later dest
    let result := backward realOps c f.api.saved (tensor s .forward (dest .forward))
      (tensor s .normalizer (dest .normalizer)) bTrees bScratch bOutput bks bhs ready
    f.denominator=some (tensor s .normalizer den) ∧ result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => normalizedLoss s.T s.K s.V (selectedScale s.K s.useScale) eps
      (affineInputs (inputValues realOps s x entry bh) dx a) (viewRead realOps s s.V dy later bh))
      (result.returned.pairing s bh dx result.memory) 0 := by
  let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false true
  let ready := propagateRun realOps s true tree f.api.rawOutput (tensor s .normalizer den) dy later dest
  let do_ := tensor s .forward (dest .forward)
  let dz := tensor s .normalizer (dest .normalizer)
  let target := normalizedCotangent s.T s.K s.V (selectedScale s.K s.useScale) eps
    (inputValues realOps s x entry bh) (viewRead realOps s s.V dy later bh)
  have raw := forward_retained c s domain x fTrees fScratch fOutput fks fhs fLayout entry eps den dst fresh bh hb
  have denValue (t : Nat) (ht : t<s.T) : (tensor s .normalizer den).read later bh t 0=
      normalizerSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t+eps :=
    (preserved.2 bh t hb ht).trans (raw.2.2 t ht)
  have oValue (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) : f.api.rawOutput.read later bh t e=
      outputSpec s.T s.K (selectedScale s.K s.useScale) (inputValues realOps s x entry bh) t e :=
    (preserved.1 bh t e hb ht he).trans (raw.2.1 t ht e he)
  have valid : SavedValid (save realOps s x) entry ready := by
    intro address ha
    calc
      ready address=later address := propagate_frame _ _ _ _ _ _ _ _ _ _
        (by intro ev he eq;exact inputFresh ev he (eq ▸ ha))
      _=entry address := saved ha
  have hx := saved_inputs realOps (save realOps s x) entry ready valid do_ dz bh
  change headInputs (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh=
    inputValues realOps s x entry bh at hx
  have ho (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) :
      (headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh).output t e=
        target.output t e := by
    change tensorRead realOps ready (dest .forward) (s.B*s.H) s.T s.V bh t e=_
    rw [tensorRead,if_pos ⟨hb,ht,he⟩]
    dsimp only [ready]
    rw [propagate_output realOps s true tree f.api.rawOutput (tensor s .normalizer den) dy later dest propLayout propFresh bh t e hb ht he]
    simp only [propagated,if_true,hb,ht,he,and_self,realOps]
    rw [denValue t ht]
    rfl
  have hz (t : Nat) (ht : t<s.T) :
      (headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh).normalizer t=
        target.normalizer t := by
    change tensorRead realOps ready (dest .normalizer) (s.B*s.H) s.T 1 bh t 0=_
    rw [tensorRead,if_pos ⟨hb,ht,by omega⟩]
    dsimp only [ready]
    rw [propagate_normalizer realOps s true tree f.api.rawOutput (tensor s .normalizer den) dy later dest propLayout propFresh bh t hb ht,
      propagated_real s tree complete _ _ _ _ _ _ hb ht,denValue t ht]
    change -(∑ e∈Finset.range s.V,_*_)/_=-(∑ e∈Finset.range s.V,_*_)/_
    congr 2
    apply Finset.sum_congr rfl
    intro e he
    rw [oValue t ht e (Finset.mem_range.mp he)]
  have h := returned_vjp c.plans s domain (backwardBases (save realOps s x) do_ dz)
    bTrees bScratch bOutput bks bhs bLayout ready bh hb dx
  rw [hx] at h
  have lossEq (a : ℝ) := loss_cotangent_congr s.T s.K s.V (selectedScale s.K s.useScale)
    (affineInputs (inputValues realOps s x entry bh) dx a) _ target ho hz
  simp_rw [lossEq] at h
  refine ⟨rfl,rfl,?_⟩
  simp only [forward,backward_real,HostAPI.backward,saved_ops_eq]
  change HasDerivAt (fun a : ℝ => normalizedLoss s.T s.K s.V (selectedScale s.K s.useScale) eps
    (affineInputs (inputValues realOps s x entry bh) dx a) (viewRead realOps s s.V dy later bh))
    (returnedPairing s bh dx bOutput
    (HostAPI.dispatch realOps c.plans s (backwardBases (save realOps s x) do_ dz)
      bTrees bScratch bOutput bks bhs ready)) 0
  rw [HostAPI.dispatch_eq _ _ _ _ _ _ _ backwardKinds _ _ bLayout]
  have same := h.unique (independent_vjp s.T s.K s.V (selectedScale s.K s.useScale)
    (inputValues realOps s x entry bh) dx target)
  rw [same]
  exact normalized_vjp s.T s.K s.V (selectedScale s.K s.useScale) eps
    (inputValues realOps s x entry bh) dx (viewRead realOps s s.V dy later bh) nonzero

/-- Later public single-output backward follows the actual optional retained
denominator handle. `getD` is unreachable in the normalized branch; the raw
branch has no denominator reads. -/
theorem single_normalized_vjp (c : CompilerLayout) (s : Shape) (domain : ShapeContract s)
    (x : HostAPI.Inputs) (metadata : x.Valid s)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (fLayout : RunLayout s (forwardBases x) fTrees fScratch fOutput forwardKinds fks fhs)
    (entry later : Memory ℝ) (eps : ℝ) (den dst : Nat) (fresh : NormalizeLayout s fOutput den dst)
    (dy : View) (tree : Nat → Nat → SumPlan) (complete : ∀ bh t,(tree bh t).Covers s.V)
    (preserved : let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false true
      RawSaved s f.api.rawOutput (tensor s .normalizer den) f.api.memory later)
    (dest : OutputKind → Nat) (propLayout : PropagationLayout s dest)
    (propFresh : PropagationFresh s (tensor s .forward (fOutput .forward)) (tensor s .normalizer den) dy dest)
    (saved : SavedValid (save realOps s x) entry later)
    (inputFresh : ∀ ev∈events s false forwardKinds,tensorAddress dest ev∉inputCellsOnly s x)
    (bTrees : OutputKind → SumPlan) (bScratch bOutput : OutputKind → Nat) (bks bhs : List Event)
    (bLayout : RunLayout s (backwardBases (save realOps s x)
      (tensor s .forward (dest .forward)) (tensor s .normalizer (dest .normalizer)))
      bTrees bScratch bOutput backwardKinds bks bhs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ)
    (nonzero : ∀ t,t<s.T → normalizerSpec s.T s.K (selectedScale s.K s.useScale)
      (inputValues realOps s x entry bh) t+eps≠0) :
    let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false true
    let result := singleBackward realOps c f true tree dy later dest bTrees bScratch bOutput bks bhs
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => normalizedLoss s.T s.K s.V (selectedScale s.K s.useScale) eps
      (affineInputs (inputValues realOps s x entry bh) dx a) (viewRead realOps s s.V dy later bh))
      (result.returned.pairing s bh dx result.memory) 0 :=
  (normalized_backward_vjp c s domain x metadata fTrees fScratch fOutput fks fhs fLayout
    entry later eps den dst fresh dy tree complete preserved dest propLayout propFresh saved inputFresh
    bTrees bScratch bOutput bks bhs bLayout bh hb dx nonzero).2

/-- Raw single-output calls materialize zero dZ and preserve the incoming
dy elements. Its public VJP has no normalization-domain condition. -/
theorem single_raw_vjp (c : CompilerLayout) (s : Shape) (domain : ShapeContract s)
    (x : HostAPI.Inputs) (metadata : x.Valid s)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (fLayout : RunLayout s (forwardBases x) fTrees fScratch fOutput forwardKinds fks fhs)
    (entry later : Memory ℝ) (eps : ℝ) (den dst : Nat) (dy : View) (tree : Nat → Nat → SumPlan)
    (dest : OutputKind → Nat) (propLayout : PropagationLayout s dest)
    (propFresh : PropagationFresh s (tensor s .forward (fOutput .forward))
      (tensor s .normalizer (fOutput .normalizer)) dy dest)
    (saved : SavedValid (save realOps s x) entry later)
    (inputFresh : ∀ ev∈events s false forwardKinds,tensorAddress dest ev∉inputCellsOnly s x)
    (bTrees : OutputKind → SumPlan) (bScratch bOutput : OutputKind → Nat) (bks bhs : List Event)
    (bLayout : RunLayout s (backwardBases (save realOps s x)
      (tensor s .forward (dest .forward)) (tensor s .normalizer (dest .normalizer)))
      bTrees bScratch bOutput backwardKinds bks bhs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ) :
    let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false false
    let result := singleBackward realOps c f false tree dy later dest bTrees bScratch bOutput bks bhs
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V (selectedScale s.K s.useScale)
      (affineInputs (inputValues realOps s x entry bh) dx a)
      (outputOnlyCotangent (viewRead realOps s s.V dy later bh)))
      (result.returned.pairing s bh dx result.memory) 0 := by
  let f := forward realOps c s x fTrees fScratch fOutput fks fhs entry eps den dst false false
  let ready := propagateRun realOps s false tree f.api.rawOutput f.api.rawNormalizer dy later dest
  let do_ := tensor s .forward (dest .forward)
  let dz := tensor s .normalizer (dest .normalizer)
  have valid : SavedValid (save realOps s x) entry ready := by
    intro address ha
    calc
      ready address=later address := propagate_frame _ _ _ _ _ _ _ _ _ _
        (by intro ev he eq;exact inputFresh ev he (eq ▸ ha))
      _=entry address := saved ha
  have hx := saved_inputs realOps (save realOps s x) entry ready valid do_ dz bh
  change headInputs (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh=
    inputValues realOps s x entry bh at hx
  have ho (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) :
      (headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh).output t e=
        viewRead realOps s s.V dy later bh t e := by
    change tensorRead realOps ready (dest .forward) (s.B*s.H) s.T s.V bh t e=_
    rw [tensorRead,if_pos ⟨hb,ht,he⟩]
    exact propagate_output realOps s false tree f.api.rawOutput f.api.rawNormalizer dy later dest propLayout propFresh bh t e hb ht he
  have hz (t : Nat) (ht : t<s.T) :
      (headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh).normalizer t=0 := by
    change tensorRead realOps ready (dest .normalizer) (s.B*s.H) s.T 1 bh t 0=_
    rw [tensorRead,if_pos ⟨hb,ht,by omega⟩]
    exact propagate_normalizer realOps s false tree f.api.rawOutput f.api.rawNormalizer dy later dest propLayout propFresh bh t hb ht
  have h := returned_vjp c.plans s domain (backwardBases (save realOps s x) do_ dz)
    bTrees bScratch bOutput bks bhs bLayout ready bh hb dx
  rw [hx] at h
  have lossEq (a : ℝ) := loss_cotangent_congr s.T s.K s.V (selectedScale s.K s.useScale)
    (affineInputs (inputValues realOps s x entry bh) dx a) _
    (outputOnlyCotangent (viewRead realOps s s.V dy later bh)) ho hz
  simp_rw [lossEq] at h
  refine ⟨rfl,?_⟩
  simp only [singleBackward,forward,Bool.false_eq_true,if_false,Option.getD_none,backward_real,
    HostAPI.backward,saved_ops_eq]
  change HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V (selectedScale s.K s.useScale)
    (affineInputs (inputValues realOps s x entry bh) dx a) (outputOnlyCotangent (viewRead realOps s s.V dy later bh)))
    (returnedPairing s bh dx bOutput (HostAPI.dispatch realOps c.plans s (backwardBases (save realOps s x) do_ dz)
      bTrees bScratch bOutput bks bhs ready)) 0
  rw [HostAPI.dispatch_eq _ _ _ _ _ _ _ backwardKinds _ _ bLayout]
  exact h


end
end VeriTile.Bench.Repairs.ParallelAttention.HostNative
