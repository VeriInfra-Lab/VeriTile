import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeGrid
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeCoverage

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def logicalRow (s : State ℝ) (X : RegionName) (D row : Nat) : List ℝ :=
  (List.range D).map (fun col => (s.mem X (col + D * row)).read .real)

def logicalMean (s : State ℝ) (X : RegionName) (D row : Nat) : ℝ :=
  (logicalRow s X D row).sum / D

def logicalVariance (s : State ℝ) (X : RegionName) (D row : Nat) : ℝ :=
  ((logicalRow s X D row).map (fun x => (x - logicalMean s X D row) ^ 2)).sum / D

def logicalInverse (s : State ℝ) (X : RegionName) (D row : Nat) : ℝ :=
  1 / Real.sqrt (logicalVariance s X D row + (1e-05 : ℝ))

def logicalOutput (s : State ℝ) (X W Bias : RegionName) (D row col : Nat) : ℝ :=
  ((s.mem X (col + D * row)).read .real - logicalMean s X D row) *
    logicalInverse s X D row * (s.mem W col).read .real + (s.mem Bias col).read .real

theorem row_real_mean_m2 (rest : Algebra ℝ) (s : State ℝ) (X : RegionName)
    (D B row : Nat) (hD : 0 < D) (hB : 0 < B) (tree : ReductionTree B) (ht : tree.Complete) :
    (rowValue (ideal rest) realEqual (atRow s row) X D B tree).mean = logicalMean s X D row ∧
    (rowValue (ideal rest) realEqual (atRow s row) X D B tree).m2 =
      ((logicalRow s X D row).map (fun x => (x - logicalMean s X D row) ^ 2)).sum := by
  have hm := whole_statistics_real_moments rest (atRow s row) X D B hD hB tree ht
  have hn : (List.range D).map (fun j => ((atRow s row).mem X (j + D * (atRow s row).pids 0)).read .real) ≠ [] := by
    intro he
    have hh := congrArg List.length he
    simp at hh
    omega
  have hh := moments_result _ _ hm hn
  simpa [rowValue, logicalMean, logicalRow, atRow] using hh

theorem row_real_inverse (rest : Algebra ℝ) (s : State ℝ) (X : RegionName)
    (D B row : Nat) (hD : 0 < D) (hB : 0 < B) (tree : ReductionTree B) (ht : tree.Complete) :
    rstdValue (ideal rest) D (rowValue (ideal rest) realEqual (atRow s row) X D B tree) =
      logicalInverse s X D row := by
  have hh := (row_real_mean_m2 rest s X D B row hD hB tree ht).2
  unfold rstdValue
  rw [hh]
  rfl

theorem row_real_output (rest : Algebra ℝ) (s : State ℝ) (X W Bias : RegionName)
    (D B row col : Nat) (hD : 0 < D) (hB : 0 < B) (tree : ReductionTree B) (ht : tree.Complete) :
    rowOutput (ideal rest) realEqual s X W Bias D B row col tree = logicalOutput s X W Bias D row col := by
  have hm := (row_real_mean_m2 rest s X D B row hD hB tree ht).1
  have hr := row_real_inverse rest s X D B row hD hB tree ht
  unfold rowOutput
  rw [hm, hr]
  rfl

/-- Independent mathematical face of the same complete program. Storage casts
are interpreted ideally here; this does not assert floating-point equality. -/
theorem whole_real_grid (rest : Algebra ℝ) (s : State ℝ) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (hD : 0 < D) (hB : 0 < B) (htree : tree.Complete)
    (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final, runRows (withEquality (ideal rest) realEqual) (rowKernel X W Bias Y Mean Rstd S D B tree) (List.range S) s = some final ∧
      (∀ row, row < S → final.mem Mean row = .mk .real (logicalMean s X D row)) ∧
      (∀ row, row < S → final.mem Rstd row = .mk .real (logicalInverse s X D row)) ∧
      (∀ row, row < S → ∀ col, col < D → final.mem Y (col + D * row) = .mk .bf16 (logicalOutput s X W Bias D row col)) ∧
      (∀ q off, (∀ row, row < S → rowMiss Y Mean Rstd D row q off) → final.mem q off = s.mem q off) := by
  obtain ⟨t, ht, he⟩ := whole_grid (ideal rest) realEqual s X W Bias Y Mean Rstd S D B tree hB sep (List.range S)
  refine ⟨t, ht, ?_, ?_, ?_, ?_⟩
  · intro row hr
    simpa only [(row_real_mean_m2 rest s X D B row hD hB tree htree).1] using he.mean row (List.mem_range.mpr hr)
  · intro row hr
    simpa only [row_real_inverse rest s X D B row hD hB tree htree] using he.rstd row (List.mem_range.mpr hr)
  · intro row hr col hc
    simpa only [row_real_output rest s X W Bias D B row col hD hB tree htree] using he.output row (List.mem_range.mpr hr) col hc
  · intro q off hmiss
    exact he.frame q off (fun row hr => hmiss row (List.mem_range.mp hr))

#axiomsClean row_real_output
#axiomsClean whole_real_grid
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
