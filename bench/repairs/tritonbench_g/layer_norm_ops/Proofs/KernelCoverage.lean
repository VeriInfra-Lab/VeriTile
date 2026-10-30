import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelInputs
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelDispatch

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelInputs
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32 Admission Shared
noncomputable section
variable {α : Type} {read : Nat → α}

def Contains (views : RegionName → Option (ReadView α read)) (r : RegionName) (i : Nat) : Prop :=
  ∃ v,views r=some v ∧ i<v.tensor.metadata.numel

theorem checked_count (read : Nat → α) (t : TensorInput) (shape steps : List Nat)
    (hc : Canonical shape steps) (h : TensorContract t.argument shape steps) :
    (checkedRead read t shape steps hc h).tensor.metadata.numel=shape.prod := by
  change t.tensor.metadata.shape.prod=shape.prod
  rw [h.2.2.2.1]

theorem optional_contains (read : Nat → α) (t : Option TensorInput) (shape steps : List Nat)
    (hc : Canonical shape steps) (h : OptionalContract (optionalArgument t) shape steps)
    (present : (optionalArgument t).present=Bool.true) (i : Nat) (hi : i<shape.prod) :
    ∃ v,optionalRead read t shape steps hc h=some v ∧ i<v.tensor.metadata.numel := by
  cases t with
  | none => simp [optionalArgument,Argument.present] at present
  | some t =>
    refine ⟨checkedRead read t shape steps hc h,rfl,?_⟩
    simpa [checked_count read t shape steps hc h] using hi

def ForwardReads (views : RegionName → Option (ReadView α read)) (c : ForwardCall) (row : Nat) : Prop :=
  (∀ j,j<c.common.columns → Contains views "x" (row*c.common.columns+j)) ∧
  (∀ j,j<c.common.columns → Contains views "weight" j) ∧
  (c.common.bias.present=Bool.true → ∀ j,j<c.common.columns → Contains views "bias" j) ∧
  (c.residual.present=Bool.true → ∀ j,j<c.common.columns → Contains views "residual" (row*c.common.columns+j))

theorem forward_coverage (read : Nat → α) (c : ForwardCall) (b : ForwardBinding c)
    (h : ForwardContract c) (row : Nat) (hr : row<c.common.rows) :
    ForwardReads (forwardViews read c b h) c row := by
  have bound (j : Nat) (hj : j<c.common.columns) : row*c.common.columns+j<c.common.rows*c.common.columns := by
    have hm := Nat.mul_le_mul_right c.common.columns (Nat.succ_le_of_lt hr)
    nlinarith
  refine ⟨?_,?_,?_,?_⟩
  · intro j hj
    refine ⟨checkedRead read b.x _ _ (.matrix c.common.rows c.common.columns) (b.xArg.symm ▸ h.1.2.2.1),?_,?_⟩
    · simp [forwardViews]
    · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using bound j hj
  · intro j hj
    refine ⟨checkedRead read b.weight _ _ (.vector c.common.columns) (b.weightArg.symm ▸ h.1.2.2.2.1),?_,?_⟩
    · simp [forwardViews]
    · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using hj
  · intro hb j hj
    obtain ⟨v,hv,hc⟩ := optional_contains read b.bias _ _ (.vector c.common.columns)
      (b.biasArg.symm ▸ h.1.2.2.2.2.1) (b.biasArg.symm ▸ hb) j (by simpa using hj)
    exact ⟨v,by simpa [forwardViews] using hv,hc⟩
  · intro hd j hj
    obtain ⟨v,hv,hc⟩ := optional_contains read b.residual _ _ (.matrix c.common.rows c.common.columns)
      (b.residualArg.symm ▸ h.2.2.1) (b.residualArg.symm ▸ hd) _ (by simpa using bound j hj)
    exact ⟨v,by simpa [forwardViews] using hv,hc⟩

def BackwardReads (views : RegionName → Option (ReadView α read)) (c : BackwardCall) (row : Nat) : Prop :=
  (∀ j,j<c.common.columns → Contains views "x" (row*c.common.columns+j) ∧
    Contains views "dy" (row*c.common.columns+j) ∧ Contains views "weight" j) ∧
  Contains views "rstd" row ∧
  (c.common.rms.getD Bool.false=Bool.false → Contains views "mean" row) ∧
  (c.dresidual.present=Bool.true → ∀ j,j<c.common.columns → Contains views "dres" (row*c.common.columns+j)) ∧
  (c.common.bias.present=Bool.true → ∀ j,j<c.common.columns → Contains views "bias" j)

/-- The backward prelude reads parameters even for an idle program. -/
def ParameterReads (views : RegionName → Option (ReadView α read)) (c : Common) : Prop :=
  (∀ j,j<c.columns → Contains views "weight" j) ∧
  (c.bias.present=Bool.true → ∀ j,j<c.columns → Contains views "bias" j)

theorem backward_parameters (read : Nat → α) (c : BackwardCall) (b : BackwardBinding c)
    (h : BackwardContract c) : ParameterReads (backwardViews read c b h) c.common := by
  constructor
  · intro j hj
    refine ⟨checkedRead read b.weight _ _ (.vector c.common.columns) (b.weightArg.symm ▸ h.1.2.2.2.1),?_,?_⟩
    · simp [backwardViews]
    · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using hj
  · intro hb j hj
    obtain ⟨v,hv,hc⟩ := optional_contains read b.bias _ _ (.vector c.common.columns)
      (b.biasArg.symm ▸ h.1.2.2.2.2.1) (b.biasArg.symm ▸ hb) j (by simpa using hj)
    exact ⟨v,by simpa [backwardViews] using hv,hc⟩

theorem backward_coverage (read : Nat → α) (c : BackwardCall) (b : BackwardBinding c)
    (h : BackwardContract c) (row : Nat) (hr : row<c.common.rows) :
    BackwardReads (backwardViews read c b h) c row := by
  have bound (j : Nat) (hj : j<c.common.columns) : row*c.common.columns+j<c.common.rows*c.common.columns := by
    have hm := Nat.mul_le_mul_right c.common.columns (Nat.succ_le_of_lt hr)
    nlinarith
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro j hj
    refine ⟨?_,?_,?_⟩
    · refine ⟨checkedRead read b.x _ _ (.matrix c.common.rows c.common.columns) (b.xArg.symm ▸ h.1.2.2.1),?_,?_⟩
      · simp [backwardViews]
      · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using bound j hj
    · refine ⟨checkedRead read b.dy _ _ (.matrix c.common.rows c.common.columns) (b.dyArg.symm ▸ h.2.1),?_,?_⟩
      · simp [backwardViews]
      · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using bound j hj
    · refine ⟨checkedRead read b.weight _ _ (.vector c.common.columns) (b.weightArg.symm ▸ h.1.2.2.2.1),?_,?_⟩
      · simp [backwardViews]
      · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using hj
  · refine ⟨checkedRead read b.rstd _ _ (.vector c.common.rows) (b.rstdArg.symm ▸ h.2.2.2.2.1),?_,?_⟩
    · simp [backwardViews]
    · simpa only [checked_count,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
  · intro hm
    have hp : c.mean.present=Bool.true := by
      have ht := h.2.2.2.1
      simp only [MeanContract,hm,Bool.false_eq_true,if_false] at ht
      cases he : c.mean <;> simp_all [TensorContract,Argument.present]
    obtain ⟨v,hv,hc⟩ := optional_contains read b.mean _ _ (.vector c.common.rows)
      (b.meanArg.symm ▸ mean_optional c h.2.2.2.1) (b.meanArg.symm ▸ hp) row (by simpa using hr)
    exact ⟨v,by simpa [backwardViews] using hv,hc⟩
  · intro hd j hj
    obtain ⟨v,hv,hc⟩ := optional_contains read b.dres _ _ (.matrix c.common.rows c.common.columns)
      (b.dresArg.symm ▸ h.2.2.1) (b.dresArg.symm ▸ hd) _ (by simpa using bound j hj)
    exact ⟨v,by simpa [backwardViews] using hv,hc⟩
  · intro hb j hj
    obtain ⟨v,hv,hc⟩ := optional_contains read b.bias _ _ (.vector c.common.columns)
      (b.biasArg.symm ▸ h.1.2.2.2.2.1) (b.biasArg.symm ▸ hb) j (by simpa using hj)
    exact ⟨v,by simpa [backwardViews] using hv,hc⟩

#axiomsClean forward_coverage
#axiomsClean backward_coverage
#axiomsClean backward_parameters
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelInputs
