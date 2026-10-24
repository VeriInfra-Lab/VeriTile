import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Checked
open VeriTile.Triton FP.Structural Reuse
noncomputable section

def sourceTrials {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) : List Configuration → State α → Option (State α)
  | [], s => some s
  | c :: rest, s => do
      let t ← runRows (withEquality M eq32)
        (Source.rowKernel X W Bias Y Mean Rstd S D c.block (SourceBinding.tree c.tree)) (List.range S) s
      sourceTrials M eq32 X W Bias Y Mean Rstd S D rest t

theorem source_trials_binding {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (trials : List Configuration) (s : State α) :
    sourceTrials M eq32 X W Bias Y Mean Rstd S D trials s = runTrials M eq32 X W Bias Y Mean Rstd S D trials s := by
  induction trials generalizing s with
  | nil => rfl
  | cons c rest ih => simp only [sourceTrials, runTrials, SourceBinding.row_binding, ih]

def sourceHost {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (trials : List Configuration) (last : Configuration) : Option (Returned α) := do
  let mean := meanTensor a S
  let backing := inverseBacking a S
  let inverse := reinterpret a.inverseViewObject backing [S,1] [1,1] 0
  let output := outputTensor a S D
  let u ← sourceTrials M eq32 X.region W.region Bias.region output.region mean.region inverse.region S D trials s
  let t ← runRows (withEquality M eq32)
    (Source.rowKernel X.region W.region Bias.region output.region mean.region inverse.region S D
      last.block (SourceBinding.tree last.tree)) (List.range S) u
  return ⟨t, (output, X, mean, inverse)⟩

theorem source_host_binding {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : Tensor) (a : Allocations) (S D : Nat)
    (trials : List Configuration) (last : Configuration) :
    sourceHost M eq32 s X W Bias a S D trials last = executeHost M eq32 s X W Bias a S D trials last := by
  simp only [sourceHost, executeHost, source_trials_binding, SourceBinding.row_binding]

/-- The Python guard runs before DeviceGuard(0), allocation, reinterpretation,
autotune trials and the selected final launch. All those successful source host
operations are retained by executeHost; no copy/canonicalization is inserted. -/
def run {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (mw mb mx : Admission.Metadata) (X W Bias : Tensor)
    (a : Allocations) (trials : List Configuration) (last : Configuration) :
    Except Admission.Error (Option (Returned α)) := do
  let dims ← Admission.checkInputs (.tensor mw) (.tensor mb) (.tensor mx)
  return sourceHost M eq32 s X W Bias a dims.1 dims.2 trials last

theorem admitted {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (mw mb mx : Admission.Metadata) (X W Bias : Tensor)
    (a : Allocations) (S D : Nat) (h : Admission.Contract mw mb mx S D)
    (trials : List Configuration) (last : Configuration) :
    run M eq32 s mw mb mx X W Bias a trials last =
      .ok (executeHost M eq32 s X W Bias a S D trials last) := by
  have hc := (Admission.checkInputs_iff (.tensor mw) (.tensor mb) (.tensor mx) S D).mpr h
  simp [run, hc, source_host_binding]

#axiomsClean source_host_binding
#axiomsClean admitted
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Checked
