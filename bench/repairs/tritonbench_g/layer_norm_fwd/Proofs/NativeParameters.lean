import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeRow

/-! Independent input-only native expression target with actual encoded scalar
arguments. Infinities, NaNs and signed zero remain values of α. Finite constants
0, 1 and N retain literal conversion; no scalar is routed through Real. The
historical Real-parameter graph remains in NativeRow for its separate proofs. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeParameters
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section

def stream (f : Flags) (x : Inputs α) (M N : Nat) (p : α)
    (second : Bool) (row col : Nat) : Expr α :=
  let raw := load (if second then x.x1 else x.x) N row col
  let index := if second then M+row else row
  let scaled := if f.rowscale then Expr.mul raw (.cast32 (.input (x.rowscale index))) else raw
  if f.dropout then .selectGT (.random (x.seeds index) col) (.input p)
    (.div scaled (.sub (.literal 1) (.input p))) (.literal 0) else scaled

def combined (f : Flags) (x : Inputs α) (M N : Nat) (p : α) (row col : Nat) : Expr α :=
  let main := stream f x M N p Bool.false row col
  let two := if f.x1 then Expr.add main (stream f x M N p Bool.true row col) else main
  if f.residual then
    if f.rowscale && !f.dropout && !f.x1 then
      .fma (load x.x N row col) (.cast32 (.input (x.rowscale row))) (load x.residual N row col)
    else .add two (load x.residual N row col)
  else two

def mean (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (p : α) (row col : Nat) : Expr α :=
  if N=1 then combined f x M N p row 0
  else .div (sumExpr (red col).mean (fun j => combined f x M N p row j.val)) (.literal N)

def centered (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (p : α) (row col : Nat) : Expr α :=
  if f.rms then combined f x M N p row col
  else .sub (combined f x M N p row col) (mean red f x M N p row col)

def variance (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (p : α) (row col : Nat) : Expr α :=
  .div (squareExpr (red col).variance (fun j => if j.val<N then centered red f x M N p row j.val else .literal 0)) (.literal N)

def rstd (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (eps p : α) (row col : Nat) : Expr α :=
  .div (.literal 1) (.sqrt (if N=1 then
    .fma (centered red f x M N p row 0) (centered red f x M N p row 0) (.input eps)
    else .add (.input eps) (variance red f x M N p row col)))

def normalized (red : Nat → Whole.Reductions block) (f : Flags) (x : Inputs α) (M N : Nat) (eps p : α) (row col : Nat) : Expr α :=
  if f.rms then .mul (rstd red f x M N eps p row col) (centered red f x M N p row col)
  else .mul (centered red f x M N p row col) (rstd red f x M N eps p row col)

def y (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row col : Nat) : Expr α :=
  affine plan f.bias (normalized red f x M N eps p row col)
    (.cast32 (.input (x.weight col))) (.cast32 (.input (x.bias col)))

def y1 (red : Nat → Whole.Reductions block) (plan : AffinePlan) (f : Flags) (x : Inputs α)
    (M N : Nat) (eps p : α) (row col : Nat) : Expr α :=
  affine plan f.b1 (normalized red f x M N eps p row col)
    (.cast32 (.input (x.weight1 col))) (.cast32 (.input (x.bias1 col)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeParameters
