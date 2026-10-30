import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceGraph
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.LayerNormWelfold.KernelGraph
open VeriTile.Triton FP.Structural Reuse Native
noncomputable section

/-- Independent graph inputs are the original logical columns and the source
D/epsilon literals; no atom denotes a statistic or expected output. -/
def logicalInputs {α : Type} (M : Algebra α) (x w bias : Nat → α)
    (D row n : Nat) : α :=
  let pair := Nat.unpair n
  match pair.1 with
  | 0 => M.fromNat (some .fp32) D
  | 1 => M.literal (some .fp32) .real (1e-05)
  | 2 => x (pair.2 + D * row)
  | 3 => w pair.2
  | _ => bias pair.2

structure Agreement {α : Type} (left right : Nat → α) (D : Nat) : Prop where
  dimension : left (Nat.pair 0 0) = right (Nat.pair 0 0)
  epsilon : left (Nat.pair 1 0) = right (Nat.pair 1 0)
  x : ∀ col, col < D → left (Nat.pair 2 col) = right (Nat.pair 2 col)
  w : ∀ col, col < D → left (Nat.pair 3 col) = right (Nat.pair 3 col)
  bias : ∀ col, col < D → left (Nat.pair 4 col) = right (Nat.pair 4 col)

private theorem lane_congr (p : Primitives α) (left right : Nat → α) (D B k : Nat)
    (h : Agreement left right D) (i : Fin B) :
    eval p left (laneExpr D B k i) = eval p right (laneExpr D B k i) := by
  by_cases active : k * B + i.val < D
  · simp only [laneExpr, if_pos active, eval, h.x _ active]
  · simp only [laneExpr, if_neg active, eval]

private theorem recurrence_congr (p : Primitives α) (left right : Nat → α) (D B : Nat)
    (h : Agreement left right D) (phase : Bool) (mean : Expr)
    (hm : eval p left mean = eval p right mean) (k count : Nat) (prev : Fin B → Expr)
    (hp : ∀ i, eval p left (prev i) = eval p right (prev i)) :
    ∀ i, eval p left (recurrenceExpr phase D B mean k count prev i) =
      eval p right (recurrenceExpr phase D B mean k count prev i) := by
  induction count generalizing k prev with
  | zero => exact hp
  | succ count ih =>
    apply ih
    intro i
    cases phase <;> by_cases active : k * B + i.val < D <;>
      simp only [advanceExpr, Bool.false_eq_true, if_false, if_true,
        active, eval, hp i, hm, lane_congr p left right D B k h i]

private theorem tree_congr (p : Primitives α) (left right : Nat → α) {B : Nat}
    (values : Fin B → Expr) (tree : ReductionTree B)
    (h : ∀ i, eval p left (values i) = eval p right (values i)) :
    eval p left (treeExpr values tree) = eval p right (treeExpr values tree) := by
  induction tree with
  | leaf i => exact h i
  | node l r hl hr => simp only [treeExpr, eval, hl, hr]

private theorem total_congr (p : Primitives α) (left right : Nat → α) (D B : Nat)
    (h : Agreement left right D) (phase : Bool) (mean : Expr)
    (hm : eval p left mean = eval p right mean) (tree : ReductionTree B) :
    eval p left (totalExpr phase D B mean tree) = eval p right (totalExpr phase D B mean tree) :=
  tree_congr p left right _ tree
    (recurrence_congr p left right D B h phase mean hm 0 (blockCount D B) _ (fun _ => rfl))

theorem mean_congr (p : Primitives α) (left right : Nat → α) (D B : Nat)
    (h : Agreement left right D) (tree : ReductionTree B) :
    eval p left (meanExpr D B tree) = eval p right (meanExpr D B tree) := by
  have ht := total_congr p left right D B h Bool.false .zero rfl tree
  have hd : eval p left (dimensionExpr D) = eval p right (dimensionExpr D) := by
    by_cases one : D = 1 <;> simp [dimensionExpr, one, eval, h.dimension]
  simp only [meanExpr, eval, ht, hd]

theorem inverse_congr (p : Primitives α) (left right : Nat → α) (D B : Nat)
    (h : Agreement left right D) (tree : ReductionTree B) :
    eval p left (inverseExpr D B tree) = eval p right (inverseExpr D B tree) := by
  have ht := total_congr p left right D B h Bool.true (meanExpr D B tree)
    (mean_congr p left right D B h tree) tree
  have hd : eval p left (dimensionExpr D) = eval p right (dimensionExpr D) := by
    by_cases one : D = 1 <;> simp [dimensionExpr, one, eval, h.dimension]
  simp only [inverseExpr, eval, ht, hd, h.epsilon]

theorem output_congr (p : Primitives α) (left right : Nat → α) (D B col : Nat)
    (h : Agreement left right D) (tree : ReductionTree B) (hc : col < D) :
    eval p left (outputExpr D B col tree) = eval p right (outputExpr D B col tree) := by
  simp only [outputExpr, loadExpr, eval, mean_congr p left right D B h tree,
    inverse_congr p left right D B h tree, h.x col hc, h.w col hc, h.bias col hc]

theorem observed_agreement {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (x w bias : Nat → α) (D row : Nat)
    (hx : ∀ col, col < D → (s.mem "X" (col + D * row)).read .real = x (col + D * row))
    (hw : ∀ col, col < D → (s.mem "W" col).read .real = w col)
    (hb : ∀ col, col < D → (s.mem "Bias" col).read .real = bias col) :
    Agreement (inputs M s "X" "W" "Bias" D row) (logicalInputs M x w bias D row) D := by
  exact ⟨by simp [inputs, logicalInputs], by simp [inputs, logicalInputs],
    by simpa [inputs, logicalInputs] using hx,
    by simpa [inputs, logicalInputs] using hw,
    by simpa [inputs, logicalInputs] using hb⟩

#axiomsClean mean_congr
#axiomsClean inverse_congr
#axiomsClean output_congr
#axiomsClean observed_agreement
end
end VeriTile.Bench.Repairs.LayerNormWelfold.KernelGraph
