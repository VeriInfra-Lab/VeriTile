import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostAPI
import bench.repairs.tritonbench_g.parallel_attention.Proofs.SourceCorrect

namespace VeriTile.Bench.Repairs.ParallelAttention.HostAPI
noncomputable section
open scoped BigOperators

/-- A returned Tensor handle includes its actual storage pointer. A storage
offset is included in `base`; allocation/metadata are ordinary Tensor contracts. -/
theorem forward_correct (p : Plans) (s : Shape) (domain : ShapeContract s)
    (x : Inputs) (_metadata : x.Valid s) (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (layout : RunLayout s (forwardBases x) trees scratch output forwardKinds ks hs)
    (m : Memory ℝ) (eps : ℝ) (normalized : Nat) (returnBoth normalize : Bool)
    (fresh : normalize=true → returnBoth=false → NormalizationLayout s output normalized)
    (bh : Nat) (hb : bh<s.B*s.H) :
    let result := forward realOps p s x trees scratch output ks hs m eps normalized returnBoth normalize
    PublicResult.Agrees s.T s.V (result.returned.read result.memory bh)
      (publicSpec s.T s.K (selectedScale s.K s.useScale) eps
        (inputValues realOps s x m bh) returnBoth normalize) := by
  have correct := (sourceCorrect p s domain (forwardBases x) trees scratch output forwardKinds ks hs layout m).1
  have ho (t : Nat) (ht : t<s.T) (e : Nat) (he : e<s.V) :=
    correct .forward (by simp [forwardKinds]) bh t e hb ht he
  have hz (t : Nat) (ht : t<s.T) :=
    correct .normalizer (by simp [forwardKinds]) bh t 0 hb ht (by norm_num [OutputKind.width])
  simp only [kernelRun,OutputKind.width,mathematicalResult,logicalSpec] at ho hz
  cases returnBoth <;> cases normalize <;>
    simp only [forward,returned,forwardMemory,
      Bool.false_eq_true,if_false,if_true,Returned.read,PublicResult.Agrees,publicSpec]
  · intro t ht e he
    simpa [tensor,Tensor.read,outputMetadata,OutputKind.width,
      dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout m,
      inputValues,mathematicalResult,logicalSpec,kernelRun] using ho t ht e he
  · intro t ht e he
    change normalizeRun realOps eps (s.B*s.H) s.T s.V (output .forward) (output .normalizer) normalized
      (dispatch realOps p s (forwardBases x) trees scratch output ks hs m)
      (normalized+dense s.T s.V bh t e)=_
    rw [normalize_output _ _ _ _ _ _ _ _ _ (fresh rfl rfl) _
      (dense_bound s.T s.V (s.B*s.H) bh t e hb ht he)]
    rw [dense_normalizer_index s.T s.V bh t e he,
      dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout m]
    change combinedRun realOps p s (forwardBases x) trees scratch output ks hs m
      (output .forward+dense s.T s.V bh t e) /
      (combinedRun realOps p s (forwardBases x) trees scratch output ks hs m
        (output .normalizer+dense s.T 1 bh t 0)+eps)=_
    rw [ho t ht e he,hz t ht]
    rfl
  · constructor
    · intro t ht e he
      simpa [tensor,Tensor.read,outputMetadata,OutputKind.width,
        dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout m,
        inputValues,mathematicalResult,logicalSpec,kernelRun] using ho t ht e he
    · intro t ht
      simpa [tensor,Tensor.read,outputMetadata,OutputKind.width,
        dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout m,
        inputValues,mathematicalResult,logicalSpec,kernelRun] using hz t ht
  · constructor
    · intro t ht e he
      simpa [tensor,Tensor.read,outputMetadata,OutputKind.width,
        dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout m,
        inputValues,mathematicalResult,logicalSpec,kernelRun] using ho t ht e he
    · intro t ht
      simpa [tensor,Tensor.read,outputMetadata,OutputKind.width,
        dispatch_eq realOps p s (forwardBases x) trees scratch output forwardKinds ks hs layout m,
        inputValues,mathematicalResult,logicalSpec,kernelRun] using hz t ht

def Gradients.pairing (g : Gradients) (s : Shape) (bh : Nat)
    (dx : ParallelAttention.Inputs ℝ) (m : Memory ℝ) : ℝ :=
  (∑ t∈Finset.range s.T,∑ e∈Finset.range s.K,dx.q t e*g.q.read m bh t e)+
  (∑ t∈Finset.range s.T,∑ e∈Finset.range s.K,dx.k t e*g.k.read m bh t e)+
  (∑ t∈Finset.range s.T,∑ e∈Finset.range s.V,dx.v t e*g.v.read m bh t e)

/-- A separate backward call uses the actual saved handles, the saved scale,
and current contiguous cotangent buffers. It returns the three actual gradient
handles and the fourth autograd slot `None`. -/
theorem backward_vjp (p : Plans) (s : Shape) (domain : ShapeContract s) (x : Inputs) (_metadata : x.Valid s)
    (forwardEntry incoming ready : Memory ℝ) (valid : SavedValid (save realOps s x) forwardEntry ready)
    (doView dzView : View) (do_ dz : Tensor)
    (ho : Materialized s .forward incoming ready doView do_)
    (hz : Materialized s .normalizer incoming ready dzView dz)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s (backwardBases (save realOps s x) do_ dz) trees scratch output backwardKinds ks hs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ) :
    let result := backward realOps p (save realOps s x) do_ dz trees scratch output ks hs ready
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V (save realOps s x).scale
      (affineInputs (inputValues realOps s x forwardEntry bh) dx a)
      (incomingCotangent realOps s incoming doView dzView bh))
      (result.returned.pairing s bh dx result.memory) 0 := by
  constructor
  · rfl
  · have h := returned_vjp p s domain (backwardBases (save realOps s x) do_ dz)
      trees scratch output ks hs layout ready bh hb dx
    have hx := saved_inputs realOps (save realOps s x) forwardEntry ready valid do_ dz bh
    have hg := materialized_cotangent realOps (save realOps s x) incoming ready doView dzView do_ dz ho hz bh
    change headInputs (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh=
      inputValues realOps s x forwardEntry bh at hx
    change headCotangent (currentInputs realOps s (backwardBases (save realOps s x) do_ dz) ready) bh=
      incomingCotangent realOps s incoming doView dzView bh at hg
    rw [hx,hg] at h
    simp only [backward,saved_ops_eq]
    change HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V (selectedScale s.K s.useScale)
      (affineInputs (inputValues realOps s x forwardEntry bh) dx a)
      (incomingCotangent realOps s incoming doView dzView bh))
      (returnedPairing s bh dx output
        (dispatch realOps p s (backwardBases (save realOps s x) do_ dz)
          trees scratch output ks hs ready)) 0
    rw [dispatch_eq realOps p s (backwardBases (save realOps s x) do_ dz)
      trees scratch output backwardKinds ks hs layout ready]
    exact h

theorem returned_context_vjp (p : Plans) (s : Shape) (domain : ShapeContract s) (x : Inputs) (metadata : x.Valid s)
    (fTrees : OutputKind → SumPlan) (fScratch fOutput : OutputKind → Nat) (fks fhs : List Event)
    (entry : Memory ℝ) (eps : ℝ) (normalized : Nat) (returnBoth normalize : Bool)
    (incoming ready : Memory ℝ) (valid : SavedValid (save realOps s x) entry ready)
    (doView dzView : View) (do_ dz : Tensor)
    (ho : Materialized s .forward incoming ready doView do_)
    (hz : Materialized s .normalizer incoming ready dzView dz)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (ks hs : List Event)
    (layout : RunLayout s (backwardBases (save realOps s x) do_ dz) trees scratch output backwardKinds ks hs)
    (bh : Nat) (hb : bh<s.B*s.H) (dx : ParallelAttention.Inputs ℝ) :
    let f := forward realOps p s x fTrees fScratch fOutput fks fhs entry eps normalized returnBoth normalize
    let result := backward realOps p f.saved do_ dz trees scratch output ks hs ready
    result.returned.scale=none ∧
    HasDerivAt (fun a : ℝ => lossSpec s.T s.K s.V f.saved.scale
      (affineInputs (inputValues realOps s x entry bh) dx a)
      (incomingCotangent realOps s incoming doView dzView bh))
      (result.returned.pairing s bh dx result.memory) 0 :=
  backward_vjp p s domain x metadata entry incoming ready valid doView dzView do_ dz ho hz
    trees scratch output ks hs layout bh hb dx


end
end VeriTile.Bench.Repairs.ParallelAttention.HostAPI
