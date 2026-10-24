import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPValues
import VeriTile.Triton.Float.TermModel

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.FPScalarTree
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section

/-- A typed, input-independent source schedule. Precision, the full tile,
ax, keepDims and output position select the plan. The compiler must bind
this structural graph; no whole reduction value equality is a numeric port. -/
abbrev Plans := FloatDType → FP.Equational.Schedules

/-- Evaluate the existing explicit tree using only a scalar add and literal.
Every padding zero is retained; no associativity or zero identity is used. -/
def eval (add : α → α → α) (zero : α) (xs : Fin n → α) : ReductionTree n → α
  | .input i => xs i
  | .zero => zero
  | .add a b => add (eval add zero xs a) (eval add zero xs b)

def inputs (dtype : FloatDType) (shape : TileShape) (ax : Fin shape.length)
    (keep : Bool) (xs : Values α dtype.toTileDType shape) :
    TileIndex (TileShape.reduceShape shape ax keep) →
      Fin (TileShape.axisDim shape ax) → α :=
  match keep with
  | Bool.false => fun i k => toFloat dtype (xs (TileShape.insertAxisIndex shape ax i k))
  | Bool.true => fun i k => toFloat dtype (xs (TileShape.replaceAxisIndex shape ax i k))

/-- The formerly abstract typed reduction is instantiated with an explicit
tree of source-precision scalar additions. It still receives the full tile. -/
def reduction (M : Algebra α) (plans : Plans) : FPValues.Reduction α where
  apply := fun p dtype {shape} ax keep xs i => ofFloat dtype
    (eval (M.binary p dtype .add) (M.literal p dtype 0)
      (inputs dtype shape ax keep xs i) (plans dtype p shape ax keep i).tree)

/-- Actual scalar-node correspondence only. Each add premise mentions that
node's two evaluated actual operands, not the complete tree's result. -/
def Local (actual reference : α → α → α) (zeroActual zeroReference : α)
    (xsActual xsReference : Fin n → α) : ReductionTree n → Prop
  | .input i => xsActual i = xsReference i
  | .zero => zeroActual = zeroReference
  | .add a b => Local actual reference zeroActual zeroReference xsActual xsReference a ∧
      Local actual reference zeroActual zeroReference xsActual xsReference b ∧
      actual (eval actual zeroActual xsActual a) (eval actual zeroActual xsActual b) =
        reference (eval actual zeroActual xsActual a) (eval actual zeroActual xsActual b)

theorem eval_eq (actual reference : α → α → α) (zeroActual zeroReference : α)
    (xsActual xsReference : Fin n → α) (tree : ReductionTree n)
    (nodes : Local actual reference zeroActual zeroReference xsActual xsReference tree) :
    eval actual zeroActual xsActual tree = eval reference zeroReference xsReference tree := by
  induction tree with
  | input i => exact nodes
  | zero => exact nodes
  | add a b ia ib =>
    change actual (eval actual zeroActual xsActual a) (eval actual zeroActual xsActual b) = _
    rw [nodes.2.2, ia nodes.1, ib nodes.2.1]
    rfl

/-- The actual primary source has FP16 reduction joins despite its surrounding
FP32 precision annotation. All BK inputs, including masked products, remain. -/
theorem primary_row (M : Algebra α) (plans : Plans) (h : Call)
    (xs : Values α .fp16 [256,h.BK]) (row : Fin 256) :
    (reduction M plans).apply (some .fp32) .fp16 ⟨1,by simp⟩ Bool.false xs (row,PUnit.unit) =
      eval (M.binary (some .fp32) .fp16 .add) (M.literal (some .fp32) .fp16 0)
        (fun k => xs (row,k,PUnit.unit))
        (plans .fp16 (some .fp32) [256,h.BK] ⟨1,by simp⟩ Bool.false (row,PUnit.unit)).tree := by
  rfl

theorem primary_leaves (plans : Plans) (h : Call) (row : Fin 256) :
    let plan := plans .fp16 (some .fp32) [256,h.BK] ⟨1,by simp⟩ Bool.false (row,PUnit.unit)
    plan.tree.leaves.Perm ((List.finRange h.BK).map some ++ List.replicate plan.padding none) := by
  exact (plans .fp16 (some .fp32) [256,h.BK] ⟨1,by simp⟩ Bool.false (row,PUnit.unit)).valid

#axiomsClean eval_eq
#axiomsClean primary_row
#axiomsClean primary_leaves
#specNonCircular Local avoiding [reduction,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.FPScalarTree
