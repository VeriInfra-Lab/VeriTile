import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeNormalizeLoop

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Inputs may alias one another. Each allocated output is fresh. -/
structure Separation (X W Bias Y Mean Rstd : RegionName) : Prop where
  mean_rstd : Mean ≠ Rstd
  mean_y : Mean ≠ Y
  rstd_y : Rstd ≠ Y
  inputs : ∀ q, q = X ∨ q = W ∨ q = Bias → q ≠ Mean ∧ q ≠ Rstd ∧ q ≠ Y

def rowValue {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X : RegionName) (D B : Nat) (tree : ReductionTree B) : WelfordTriple α :=
  treeValue M eq32 (allLaneValues M s X D B) tree

def rowMiss (Y Mean Rstd : RegionName) (D row : Nat) (q : RegionName) (off : Nat) : Prop :=
  (q ≠ Mean ∨ off ≠ row) ∧ (q ≠ Rstd ∨ off ≠ row) ∧
    (q ≠ Y ∨ ∀ col, col < D → off ≠ col + D * row)

structure RowEffect {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) : Prop where
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  mean : t.mem Mean (s.pids 0) = .mk .real (rowValue M eq32 s X D B tree).mean
  rstd : t.mem Rstd (s.pids 0) = .mk .real (rstdValue M D (rowValue M eq32 s X D B tree))
  output : ∀ col, col < D → t.mem Y (col + D * s.pids 0) =
    .mk .bf16 (normalizeValue M s X W Bias D (s.pids 0) col
      (rowValue M eq32 s X D B tree).mean (rstdValue M D (rowValue M eq32 s X D B tree)))
  frame : ∀ q off, rowMiss Y Mean Rstd D (s.pids 0) q off → t.mem q off = s.mem q off

theorem statistics_frame {α : Type} (M : Algebra α) (s : State α)
    (Mean Rstd : RegionName) (D : Nat) (v : WelfordTriple α) (q : RegionName) (off : Nat)
    (hm : q ≠ Mean ∨ off ≠ s.pids 0) (hr : q ≠ Rstd ∨ off ≠ s.pids 0) :
    (statisticsMemory M s Mean Rstd D v).mem q off = s.mem q off := by
  unfold statisticsMemory
  rw [State.write_other _ _ _ _ _ _ hr, State.write_other _ _ _ _ _ _ hm]

end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
