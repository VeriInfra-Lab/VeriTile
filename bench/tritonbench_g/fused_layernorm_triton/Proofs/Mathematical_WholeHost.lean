import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeTrials
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeResult

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

#axiomsClean inverse_view_alias
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
