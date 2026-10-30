import bench.tritonbench_g.ksoftmax_triton.Proofs.GridValue

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

theorem maskedValue_congr {α : Type} (M : Algebra α) (fallback : α) (K row : Nat)
    (causal : Bool) {D : Nat} (x y : Fin D → α)
    (h : ∀ j, Forward.readActive K row causal j.val=Bool.true → x j=y j) :
    maskedValue M fallback K row causal x=maskedValue M fallback K row causal y := by
  funext j
  unfold maskedValue
  by_cases hj : Forward.readActive K row causal j.val=Bool.true <;> simp [hj,h]

theorem selectedValue_congr {α : Type} (M : Algebra α) (fallback : α) (K row : Nat)
    (causal : Bool) {D : Nat} (x y : Fin D → α)
    (h : ∀ j, Forward.readActive K row causal j.val=Bool.true → x j=y j) :
    selectedValue M fallback K row causal x=selectedValue M fallback K row causal y := by
  funext j
  unfold selectedValue
  rw [maskedValue_congr M fallback K row causal x y h]

theorem forwardValue_congr {α : Type} (M : Algebra α) (K row : Nat) (log causal : Bool)
    (mode : Forward.MaskMode) {D : Nat} (x x' m m' : Fin D → α)
    (hx : ∀ j, Forward.readActive K row causal j.val=Bool.true → x j=x' j)
    (hm : ∀ j, mode.present=Bool.true → Forward.readActive K row causal j.val=Bool.true → m j=m' j)
    (i : Fin D) : forwardValue M K row log causal mode x m i=forwardValue M K row log causal mode x' m' i := by
  dsimp only [forwardValue]
  rw [selectedValue_congr M M.negInf K row causal x x' hx]
  by_cases present : mode.present=Bool.true
  · rw [maskedValue_congr M M.negInf K row causal m m' (fun j h => hm j present h)]
  · simp [present]

theorem backwardValue_congr {α : Type} (M : Algebra α) (K row : Nat) (log causal : Bool)
    {D : Nat} (g g' o o' : Fin D → α)
    (hg : ∀ j, Forward.readActive K row causal j.val=Bool.true → g j=g' j)
    (ho : ∀ j, Forward.readActive K row causal j.val=Bool.true → o j=o' j)
    (i : Fin D) : backwardValue M K row log causal g o i=backwardValue M K row log causal g' o' i := by
  dsimp only [backwardValue]
  rw [selectedValue_congr M _ K row causal g g' hg,selectedValue_congr M _ K row causal o o' ho]

def Operation.causal : Operation → Bool
  | .forward _ c _ | .backward _ c => c

def Operation.readsRight : Operation → Bool
  | .forward _ _ m => m.present | .backward _ _ => Bool.true

theorem gridValue_congr {α : Type} (M : Algebra α) (op : Operation) (Q K batch row : Nat)
    {D : Nat} (left left' right right' : Nat → α)
    (hl : ∀ j : Fin D, Forward.readActive K row op.causal j.val=Bool.true →
      left (Native.flat Q K batch row j.val)=left' (Native.flat Q K batch row j.val))
    (hr : ∀ j : Fin D, op.readsRight=Bool.true → Forward.readActive K row op.causal j.val=Bool.true →
      right (rightAddress op Q K batch row j.val)=right' (rightAddress op Q K batch row j.val))
    (i : Fin D) : gridValue M op Q K batch row left right i=gridValue M op Q K batch row left' right' i := by
  cases op with
  | forward log causal mode => exact forwardValue_congr M K row log causal mode _ _ _ _ hl hr i
  | backward log causal => exact backwardValue_congr M K row log causal _ _ _ _ hl (fun j h => hr j rfl h) i

#axiomsClean gridValue_congr
end
end VeriTile.Bench.Repairs.Ksoftmax
