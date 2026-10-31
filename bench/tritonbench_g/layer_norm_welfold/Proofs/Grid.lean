import bench.tritonbench_g.layer_norm_welfold.Proofs.Row

namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem mean_congr {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X : RegionName) (D B row : Nat) (tree : ReductionTree B) (hm : s.mem X=t.mem X) :
    meanValue M (atRow s row) X D B tree = meanValue M (atRow t row) X D B tree := by
  unfold meanValue allLanes
  rw [recurrence_congr M Bool.false (atRow s row) (atRow t row) X D B 0 (blockCount D B) _ _ hm rfl]

theorem inverse_congr {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X : RegionName) (D B row : Nat) (tree : ReductionTree B) (hm : s.mem X=t.mem X) :
    inverseValue M (atRow s row) X D B tree = inverseValue M (atRow t row) X D B tree := by
  unfold inverseValue
  rw [mean_congr M s t X D B row tree hm]
  unfold allLanes
  rw [recurrence_congr M Bool.true (atRow s row) (atRow t row) X D B 0 (blockCount D B) _ _ hm rfl]

def rowOutput {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (X W Bias : RegionName) (D B row col : Nat) (tree : ReductionTree B) : α :=
  normalizeValue M s X W Bias D row col (meanValue M (atRow s row) X D B tree)
    (inverseValue M (atRow s row) X D B tree)

structure GridEffect {α : Type} [Inhabited α] (M : Algebra α)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) (rows : List Nat) : Prop where
  mean : ∀ row ∈ rows, t.mem Mean row = .mk .real (meanValue M (atRow s row) X D B tree)
  rstd : ∀ row ∈ rows, t.mem Rstd row = .mk .real (inverseValue M (atRow s row) X D B tree)
  output : ∀ row ∈ rows, ∀ col, col < D → t.mem Y (col + D * row) =
    .mk .bf16 (rowOutput M s X W Bias D B row col tree)
  frame : ∀ q off, (∀ row ∈ rows, rowMiss Y Mean Rstd D row q off) → t.mem q off = s.mem q off

end
end VeriTile.Bench.Repairs.LayerNormWelfold
