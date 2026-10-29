import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.LegacySource
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeHost

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

/-- The supplied expressions are the entire source-selected ordered helper
expansion. Its completeness is structural, never a whole-reduction FP axiom. -/
def kernel (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration) : ComputeKernel :=
  LegacySource.fusedKernel X W Bias Y Mean Rstd S D c.block
    (treeExpressions c.tree).mean (treeExpressions c.tree).m2 (treeExpressions c.tree).weight

theorem chunks (X : RegionName) (D B start count : Nat) :
    LegacySource.chunkCode X D B start count = chunkCode X D B start count := by
  induction count generalizing start with
  | zero => rfl
  | succ n ih => simp only [LegacySource.chunkCode, chunkCode, ih]; rfl

theorem normalization (X W Bias Y : RegionName) (D B start count : Nat) :
    LegacySource.normalizeCode X W Bias Y D B start count = normalizeCode X W Bias Y D B start count := by
  induction count generalizing start with
  | zero => rfl
  | succ n ih => simp only [LegacySource.normalizeCode, normalizeCode, ih]; rfl

theorem kernel_eq (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration) :
    kernel X W Bias Y Mean Rstd S D c = rowKernel X W Bias Y Mean Rstd S D c.block c.tree := by
  unfold kernel LegacySource.fusedKernel rowKernel statisticsKernel
  simp only [ComputeKernel.surfaceBody, chunks, normalization]
  rfl

def trials {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) : List Configuration → State α → Option (State α)
  | [], s => some s
  | c :: rest, s => do
      let t ← runRows (withEquality M eq32) (kernel X W Bias Y Mean Rstd S D c) (List.range S) s
      trials M eq32 X W Bias Y Mean Rstd S D rest t

theorem trials_eq {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (cs : List Configuration) (s : State α) :
    trials M eq32 X W Bias Y Mean Rstd S D cs s = runTrials M eq32 X W Bias Y Mean Rstd S D cs s := by
  induction cs generalizing s with
  | nil => rfl
  | cons c rest ih => simp only [trials, runTrials, kernel_eq, ih]

def host {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (cs : List Configuration) (last : Configuration) : Option (Returned α) := do
  let mean := meanTensor a S
  let backing := inverseBacking a S
  let inverse := reinterpret a.inverseViewObject backing [S,1] [1,1] 0
  let output := outputTensor a S D
  let u ← trials M eq32 X.region W.region Bias.region output.region mean.region inverse.region S D cs s
  let t ← runRows (withEquality M eq32)
    (kernel X.region W.region Bias.region output.region mean.region inverse.region S D last) (List.range S) u
  return ⟨t,(output,X,mean,inverse)⟩

theorem host_eq {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (cs : List Configuration) (last : Configuration) :
    host M eq32 s X W Bias a S D cs last = executeHost M eq32 s X W Bias a S D cs last := by
  simp only [host, executeHost, trials_eq, kernel_eq]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Source
