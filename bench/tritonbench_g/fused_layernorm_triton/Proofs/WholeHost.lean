import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeTrials
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeResult

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

inductive StorageDType where
  | fp32 | bf16
  deriving DecidableEq, Repr

/-- Object token distinguishes returning the original Tensor from reconstructing
an equivalent view. Regions denote backing storage, permitting input aliasing. -/
structure Tensor where
  objectId : Nat
  region : RegionName
  shape : List Nat
  strides : List Nat
  dtype : StorageDType
  storageOffset : Nat
  device : Nat
  deriving DecidableEq, Repr

structure Allocations where
  mean : RegionName
  inverse : RegionName
  output : RegionName
  meanObject : Nat
  inverseObject : Nat
  inverseViewObject : Nat
  outputObject : Nat

/-- Ordinary allocator contract. Total abstract memory already includes arbitrary
contents for fresh allocations; successful allocation preserves caller storage. -/
structure Allocations.Fresh (a : Allocations) (live : RegionName → Prop) : Prop where
  mean_inverse : a.mean ≠ a.inverse
  mean_output : a.mean ≠ a.output
  inverse_output : a.inverse ≠ a.output
  caller : ∀ q, live q → q ≠ a.mean ∧ q ≠ a.inverse ∧ q ≠ a.output

def emptyStrided (objectId : Nat) (region : RegionName) (shape strides : List Nat)
    (dtype : StorageDType) : Tensor :=
  ⟨objectId, region, shape, strides, dtype, 0, 0⟩

def reinterpret (objectId : Nat) (t : Tensor) (shape strides : List Nat) (offset : Nat) : Tensor :=
  { t with objectId := objectId, shape := shape, strides := strides, storageOffset := t.storageOffset + offset }

def meanTensor (a : Allocations) (S : Nat) : Tensor :=
  emptyStrided a.meanObject a.mean [S,1] [1,1] .fp32

def inverseBacking (a : Allocations) (S : Nat) : Tensor :=
  emptyStrided a.inverseObject a.inverse [S,1] [1,S] .fp32

def inverseTensor (a : Allocations) (S : Nat) : Tensor :=
  reinterpret a.inverseViewObject (inverseBacking a S) [S,1] [1,1] 0

def outputTensor (a : Allocations) (S D : Nat) : Tensor :=
  emptyStrided a.outputObject a.output [S,D] [D,1] .bf16

structure InputContract {α : Type} (s : State α) (X W Bias : Tensor) (S D : Nat) : Prop where
  width : 0 < D
  xMetadata : X.shape = [S,D] ∧ X.strides = [D,1] ∧ X.dtype = .fp32 ∧ X.storageOffset = 0 ∧ X.device = 0
  wMetadata : W.shape = [D] ∧ W.strides = [1] ∧ W.dtype = .fp32 ∧ W.storageOffset = 0 ∧ W.device = 0
  bMetadata : Bias.shape = [D] ∧ Bias.strides = [1] ∧ Bias.dtype = .fp32 ∧ Bias.storageOffset = 0 ∧ Bias.device = 0
  xCells : ∀ row, row < S → ∀ col, col < D → ∃ v, s.mem X.region (col + D * row) = .mk .real v
  wCells : ∀ col, col < D → ∃ v, s.mem W.region col = .mk .real v
  bCells : ∀ col, col < D → ∃ v, s.mem Bias.region col = .mk .real v

structure Returned (α : Type) where
  state : State α
  tensors : Tensor × Tensor × Tensor × Tensor

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
  let t ← runRows (withEquality M eq32)
    (rowKernel X.region W.region Bias.region output.region mean.region inverse.region S D last.block last.tree)
    (List.range S) u
  return ⟨t, (output, X, mean, inverse)⟩

def ReturnedCorrect {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (last : Configuration) (ret : Returned α) : Prop :=
  ret.tensors = (outputTensor a S D, X, meanTensor a S, inverseTensor a S) ∧
  GridEffect M eq32 s ret.state X.region W.region Bias.region a.output a.mean a.inverse D last.block last.tree (List.range S)

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

/-- The mathematical target is independent of the Welford recurrence. -/
theorem host_real_complete (rest : Algebra ℝ) (s : State ℝ) (X W Bias : Tensor)
    (a : Allocations) (S D : Nat) (domain : InputContract s X W Bias S D)
    (live : RegionName → Prop) (hX : live X.region) (hW : live W.region) (hBias : live Bias.region)
    (fresh : a.Fresh live) (trials : List Configuration) (last : Configuration) (complete : last.tree.Complete) :
    ∃ ret, executeHost (ideal rest) realEqual s X W Bias a S D trials last = some ret ∧
      ret.tensors = (outputTensor a S D, X, meanTensor a S, inverseTensor a S) ∧
      (∀ row, row < S → ret.state.mem a.mean row = .mk .real (logicalMean s X.region D row)) ∧
      (∀ row, row < S → ret.state.mem a.inverse row = .mk .real (logicalInverse s X.region D row)) ∧
      (∀ row, row < S → ∀ col, col < D → ret.state.mem a.output (col + D * row) =
        .mk .bf16 (logicalOutput s X.region W.region Bias.region D row col)) ∧
      (∀ q off, gridMiss a.output a.mean a.inverse S D q off → ret.state.mem q off = s.mem q off) := by
  obtain ⟨ret, hr, ht, hf⟩ := host_complete (ideal rest) realEqual s X W Bias a S D domain live hX hW hBias fresh trials last
  refine ⟨ret, hr, ht.1, ?_, ?_, ?_, ?_⟩
  · intro row hrow
    simpa only [(row_real_mean_m2 rest s X.region D last.block row domain.width last.positive last.tree complete).1] using ht.2.mean row (List.mem_range.mpr hrow)
  · intro row hrow
    simpa only [row_real_inverse rest s X.region D last.block row domain.width last.positive last.tree complete] using ht.2.rstd row (List.mem_range.mpr hrow)
  · intro row hrow col hc
    simpa only [row_real_output rest s X.region W.region Bias.region D last.block row col domain.width last.positive last.tree complete] using ht.2.output row (List.mem_range.mpr hrow) col hc
  · intro q off hm
    exact ht.2.frame q off (fun row hrow => hm row (List.mem_range.mp hrow))

#axiomsClean inverse_view_alias
#axiomsClean host_complete
#axiomsClean host_real_complete
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
