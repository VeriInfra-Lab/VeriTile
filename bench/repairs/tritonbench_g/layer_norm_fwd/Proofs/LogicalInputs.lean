import bench.tritonbench_g.layer_norm_fwd.Proofs.Formula
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LayerNormFwd.LogicalInputs
open VeriTile.Bench.TritonBenchG.LayerNormFwd

/-- Coordinate-preserving tensor-library resolve_neg contract for every real
input role. Strides/storage may change; values and the saved return identity do
not. These are local host materialization laws, not whole-kernel assumptions. -/
structure Resolved (logical materialized : Inputs) : Prop where
  x : ∀ i j, materialized.x i j = logical.x i j
  x1 : ∀ i j, materialized.x1 i j = logical.x1 i j
  residual : ∀ i j, materialized.residual i j = logical.residual i j
  weight : ∀ j, materialized.weight j = logical.weight j
  bias : ∀ j, materialized.bias j = logical.bias j
  weight1 : ∀ j, materialized.weight1 j = logical.weight1 j
  bias1 : ∀ j, materialized.bias1 j = logical.bias1 j
  rowscale : ∀ i, materialized.rowscale i = logical.rowscale i
  seeds : ∀ i, materialized.seeds i = logical.seeds i

theorem inputs_equal (a b : Inputs) (h : Resolved a b) : b = a := by
  cases a; cases b
  have hx := funext (fun i => funext (h.x i))
  have hx1 := funext (fun i => funext (h.x1 i))
  have hr := funext (fun i => funext (h.residual i))
  have hw := funext h.weight
  have hb := funext h.bias
  have hw1 := funext h.weight1
  have hb1 := funext h.bias1
  have hs := funext h.rowscale
  have he := funext h.seeds
  simp_all

/-- Host return saves the original input before materialization. -/
def residualReturn (original : α) (allocated : Option α) : α := allocated.getD original

theorem original_identity (x : α) : residualReturn x none = x := rfl

theorem independent_formula (a b : Inputs) (h : Resolved a b)
    (f : Flags) (M N : Nat) (eps p : ℝ) (row col : Nat) :
    Formula.y f b M N eps p row col = Formula.y f a M N eps p row col := by
  rw [inputs_equal a b h]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormFwd.LogicalInputs
