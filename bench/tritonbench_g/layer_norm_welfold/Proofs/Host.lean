import bench.tritonbench_g.layer_norm_welfold.Proofs.Trials
import bench.tritonbench_g.layer_norm_welfold.Proofs.Real

namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse Samples
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Exact source metadata: the mean allocation has strides [1,S]. -/
def meanTensor (a : Allocations) (S : Nat) : Tensor :=
  emptyStrided a.meanObject a.mean [S,1] [1,S] .fp32

/-- Successful ordinary host branch, preserving the source allocation shapes,
strides, device 0, reinterpret alias, both configs, and exact return order. -/
def executeHost {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (trials : List Configuration) (last : Configuration) : Option (Returned α) := do
  let mean := meanTensor a S
  let backing := inverseBacking a S
  let inverse := reinterpret a.inverseViewObject backing [S,1] [1,1] 0
  let output := outputTensor a S D
  let u ← runTrials M eq32 X.region W.region Bias.region output.region mean.region inverse.region S D trials s
  let t ← SourceCode.runRows (withEquality M eq32)
    (SourceCode.rowKernel X.region W.region Bias.region output.region mean.region inverse.region S D last.block (Readable.tree last.tree))
    (List.range S) u
  return ⟨t, (output, X, mean, inverse)⟩

def ReturnedCorrect {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (last : Configuration) (ret : Returned α) : Prop :=
  ret.tensors = (outputTensor a S D, X, meanTensor a S, inverseTensor a S) ∧
  GridEffect M s ret.state X.region W.region Bias.region a.output a.mean a.inverse D last.block last.tree (List.range S)

theorem inverse_view_alias (a : Allocations) (S : Nat) :
    (inverseTensor a S).region = (inverseBacking a S).region ∧
    (inverseTensor a S).dtype = .fp32 ∧ (inverseTensor a S).shape = [S,1] ∧
    (inverseTensor a S).strides = [1,1] ∧ (inverseTensor a S).storageOffset = 0 ∧
    (∀ row (col : Fin 1), row * 1 + col.val * S = row * 1 + col.val * 1) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, fun row col => statistics_reinterpret S row col⟩

theorem host_complete {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (domain : InputContract s X W Bias S D) (live : RegionName → Prop)
    (hX : live X.region) (hW : live W.region) (hBias : live Bias.region)
    (fresh : a.Fresh live) (trials : List Configuration) (last : Configuration) :
    ∃ ret, executeHost M eq32 s X W Bias a S D trials last = some ret ∧
      ReturnedCorrect M eq32 s X W Bias a S D last ret ∧
      (∀ q, live q → ∀ off, ret.state.mem q off = s.mem q off) := by
  have sep : Separation X.region W.region Bias.region a.output a.mean a.inverse := by
    refine ⟨fresh.mean_inverse, fresh.mean_output, fresh.inverse_output, ?_⟩
    intro q hq
    rcases hq with rfl | rfl | rfl
    · exact fresh.caller _ hX
    · exact fresh.caller _ hW
    · exact fresh.caller _ hBias
  obtain ⟨u, t, hu, ht, he⟩ := selected_after_trials M eq32 s X.region W.region Bias.region a.output a.mean a.inverse S D sep trials last
  let ret : Returned α := ⟨t, (outputTensor a S D, X, meanTensor a S, inverseTensor a S)⟩
  refine ⟨ret, ?_, ⟨rfl, he⟩, ?_⟩
  · simpa only [executeHost, outputTensor, meanTensor, inverseBacking, emptyStrided,
      reinterpret, bind, Option.bind, hu, ht, inverseTensor, ret, pure]
  · intro q hq off
    apply he.frame
    intro row hr
    exact ⟨.inl (fresh.caller q hq).1, .inl (fresh.caller q hq).2.1, .inl (fresh.caller q hq).2.2⟩

/-- Real erasure connects the complete returned tensor to the independent column formulas. -/
def targetOutput (s : State ℝ) (X W Bias : RegionName) (D row col : Nat) : ℝ :=
  (((s.mem X (col+D*row)).read .real-targetMean (atRow s row) X D)*
    targetInverse (atRow s row) X D)*(s.mem W col).read .real+(s.mem Bias col).read .real

theorem host_real_complete (rest : Algebra ℝ) (s : State ℝ) (X W Bias : Tensor)
    (a : Allocations) (S D : Nat) (domain : InputContract s X W Bias S D)
    (live : RegionName → Prop) (hX : live X.region) (hW : live W.region) (hBias : live Bias.region)
    (fresh : a.Fresh live) (trials : List Configuration) (last : Configuration) (complete : last.tree.Complete) :
    ∃ ret, executeHost (ideal rest) (fun _ _ => Bool.false) s X W Bias a S D trials last=some ret ∧
      ret.tensors=(outputTensor a S D,X,meanTensor a S,inverseTensor a S) ∧
      (∀ row, row<S → ret.state.mem a.mean row=.mk .real (targetMean (atRow s row) X.region D)) ∧
      (∀ row, row<S → ret.state.mem a.inverse row=.mk .real (targetInverse (atRow s row) X.region D)) ∧
      (∀ row, row<S → ∀ col, col<D → ret.state.mem a.output (col+D*row)=
        .mk .bf16 (targetOutput s X.region W.region Bias.region D row col)) ∧
      (∀ q off, gridMiss a.output a.mean a.inverse S D q off → ret.state.mem q off=s.mem q off) := by
  obtain ⟨ret,hr,ht,hf⟩ := host_complete (ideal rest) (fun _ _ => Bool.false) s X W Bias a S D domain live hX hW hBias fresh trials last
  refine ⟨ret,hr,ht.1,?_,?_,?_,?_⟩
  · intro row hrow
    simpa only [mean_real rest (atRow s row) X.region D last.block last.tree last.positive complete] using ht.2.mean row (List.mem_range.mpr hrow)
  · intro row hrow
    simpa only [inverse_real rest (atRow s row) X.region D last.block last.tree last.positive complete] using ht.2.rstd row (List.mem_range.mpr hrow)
  · intro row hrow col hc
    have ho := ht.2.output row (List.mem_range.mpr hrow) col hc
    simp only [rowOutput] at ho
    rw [mean_real rest (atRow s row) X.region D last.block last.tree last.positive complete,
      inverse_real rest (atRow s row) X.region D last.block last.tree last.positive complete] at ho
    simpa only [normalizeValue,ideal,targetOutput] using ho
  · intro q off hm
    exact ht.2.frame q off (fun row hrow => hm row (List.mem_range.mp hrow))

#axiomsClean inverse_view_alias
#axiomsClean host_complete
#axiomsClean host_real_complete
end
end VeriTile.Bench.Repairs.LayerNormWelfold
