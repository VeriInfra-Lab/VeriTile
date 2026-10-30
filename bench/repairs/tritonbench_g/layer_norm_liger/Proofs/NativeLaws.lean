import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativePlan
namespace VeriTile.Bench.Repairs.LayerNormLiger.NativePlan
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

theorem binary_ideal (input : FP.Value .fp32 → ℝ) (contract : Contraction)
    (pr : Option ComputeDType) (d : FloatDType) (op : Binary) (x y : Expr) :
    (binary contract pr d op x y).ideal input=realBinary op (x.ideal input) (y.ideal input) := by
  unfold binary
  split
  · split <;> simp_all [Expr.ideal,realBinary] <;> ring
  · rfl

theorem cast_ideal (input : FP.Value .fp32 → ℝ) (pr : Option ComputeDType)
    (src dst : FloatDType) (x : Expr) :
    (cast pr src dst x).ideal input=x.ideal input := by
  unfold cast
  split <;> rfl

/-- Every selected native reduction tree expands to its scalar leaves; local
contraction preserves the real formula, without assuming FP associativity. -/
theorem tree_ideal (input : FP.Value .fp32 → ℝ) (contract : Contraction)
    (pr : Option ComputeDType) {n : Nat} (t : ReductionPlan.Tree n) (xs : Fin n → Expr) :
    (t.eval (.literal pr .fp32 0) (binary contract pr .fp32 .add) xs).ideal input=
      t.eval 0 (·+·) (fun i => (xs i).ideal input) := by
  induction t with
  | zero => rfl
  | leaf i => rfl
  | add l r hl hr =>
    simp only [ReductionPlan.Tree.eval,binary_ideal,realBinary,hl,hr]

theorem reduction_ideal (input : FP.Value .fp32 → ℝ) (contract : Contraction)
    (pr : Option ComputeDType) {n : Nat} (p : ReductionPlan.Plan n) (xs : Fin n → Expr) :
    (p.tree.eval (.literal pr .fp32 0) (binary contract pr .fp32 .add) xs).ideal input=
      ((List.finRange n).map (fun i => (xs i).ideal input)).sum := by
  rw [tree_ideal,p.real_sum]

#axiomsClean binary_ideal
#axiomsClean cast_ideal
#axiomsClean tree_ideal
#axiomsClean reduction_ideal
end
end VeriTile.Bench.Repairs.LayerNormLiger.NativePlan
