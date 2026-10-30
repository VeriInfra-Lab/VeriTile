import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPReference
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Target

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Ideal
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section

def lanes {s : TileShape} (ax : Fin s.length) (keep : Bool)
    (xs : Values (WithBot ℝ) .real s)
    (out : TileIndex (TileShape.reduceShape s ax keep)) : Fin (TileShape.axisDim s ax) → WithBot ℝ :=
  match keep with
  | Bool.false => fun i => xs (TileShape.insertAxisIndex s ax out i)
  | Bool.true => fun i => xs (TileShape.replaceAxisIndex s ax out i)

def maximum {n : Nat} (xs : Fin n → WithBot ℝ) : WithBot ℝ := Finset.univ.sup xs

/-- Bottom is propagated by addition. This interpretation never silently drops
an exceptional summand; reached source sums are proved to contain finite terms. -/
def sum {n : Nat} (xs : Fin n → WithBot ℝ) : WithBot ℝ :=
  if ∀ i, xs i ≠ ⊥ then ((∑ i, (xs i).unbotD 0 : ℝ) : WithBot ℝ) else ⊥

def binary : Binary → WithBot ℝ → WithBot ℝ → WithBot ℝ
  | .add => WithBot.realAdd
  | .sub => WithBot.realSub
  | .mul => WithBot.realMul
  | .div => WithBot.realDiv
  | .max => fun a b => a ⊔ b
  | .pow => fun _ _ => ⊥

/-- Ideal arithmetic is used only for the stated finite and negative-infinity mathematical
scope. Native all-word outcomes use the separate instruction interpretation. -/
def algebra : Algebra (WithBot ℝ) where
  literal := fun _ _ x => (x : WithBot ℝ)
  negInf := ⊥
  binary := fun _ _ => binary
  unary := fun _ op x => if op=.exp then WithBot.realExp x else ⊥
  cast := fun _ _ _ x => x
  fromNat := fun _ n => ((n : ℝ) : WithBot ℝ)
  fromInt := fun _ n => ((n : ℝ) : WithBot ℝ)
  fp32Bits := fun _ => ⊥
  fp32Load := id
  reduceMax := fun _ {_} ax keep xs out => maximum (lanes ax keep xs out)
  reduceSum := fun _ {_} ax keep xs out => sum (lanes ax keep xs out)

@[simp] theorem sum_coe {n : Nat} (xs : Fin n → ℝ) :
    sum (fun i => (xs i : WithBot ℝ)) = ((∑ i, xs i : ℝ) : WithBot ℝ) := by
  unfold sum
  rw [if_pos (by intro i; simp)]
  rfl

theorem maximum_finite {n : Nat} (xs : Fin n → WithBot ℝ) (h : ∃ i, xs i ≠ ⊥) :
    ∃ m : ℝ, maximum xs = (m : WithBot ℝ) := by
  obtain ⟨i,hi⟩ := h
  have le : xs i ≤ maximum xs := Finset.le_sup (Finset.mem_univ _)
  cases hm : maximum xs with
  | bot =>
    rw [hm] at le
    exact (hi (le_bot_iff.mp le)).elim
  | coe m => exact ⟨m,rfl⟩

@[simp] theorem exp_value (x : WithBot ℝ) :
    FPReference.exp algebra x = (Target.mass x : WithBot ℝ) := by
  cases x <;> rfl

@[simp] theorem max_value (x y : WithBot ℝ) : FPReference.max algebra x y = x ⊔ y := rfl
@[simp] theorem literal_value (x : ℝ) : FPReference.literal algebra x = (x : WithBot ℝ) := rfl
@[simp] theorem cast_value (x : WithBot ℝ) : FPReference.cast algebra x = x := rfl
@[simp] theorem add_coe (x y : ℝ) :
    FPReference.add algebra (x : WithBot ℝ) (y : WithBot ℝ) = ((x+y : ℝ) : WithBot ℝ) := rfl
@[simp] theorem sub_coe (x y : ℝ) :
    FPReference.sub algebra (x : WithBot ℝ) (y : WithBot ℝ) = ((x-y : ℝ) : WithBot ℝ) := rfl
@[simp] theorem mul_coe (x y : ℝ) :
    FPReference.mul algebra (x : WithBot ℝ) (y : WithBot ℝ) = ((x*y : ℝ) : WithBot ℝ) := rfl
@[simp] theorem div_coe (x y : ℝ) :
    FPReference.div algebra (x : WithBot ℝ) (y : WithBot ℝ) = ((x/y : ℝ) : WithBot ℝ) := rfl

@[simp] theorem exp_sub (x : WithBot ℝ) (m : ℝ) :
    FPReference.exp algebra (FPReference.sub algebra x (m : WithBot ℝ)) =
      ((Real.exp (-m) * Target.mass x : ℝ) : WithBot ℝ) := by
  cases x with
  | bot => change ((0 : ℝ) : WithBot ℝ) = ((Real.exp (-m)*0 : ℝ) : WithBot ℝ); simp
  | coe x =>
    change ((Real.exp (x-m) : ℝ) : WithBot ℝ) = ((Real.exp (-m)*Real.exp x : ℝ) : WithBot ℝ)
    congr 1
    rw [←Real.exp_add]
    congr 1
    ring

@[simp] theorem sum_inner {B : Nat} (xs : TileIndex [B] → ℝ) :
    algebra.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
      (fun i => (xs i : WithBot ℝ)) PUnit.unit =
      ((∑ i : Fin B, xs (i,PUnit.unit) : ℝ) : WithBot ℝ) := by
  change sum (fun i : Fin B => (xs (i,PUnit.unit) : WithBot ℝ)) = _
  exact sum_coe _

@[simp] theorem sum_noninner {B C : Nat} (xs : TileIndex [B,C] → ℝ) (col : Fin C) :
    algebra.reduceSum (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false
      (fun i => (xs i : WithBot ℝ)) (col,PUnit.unit) =
      ((∑ i : Fin B, xs (i,col,PUnit.unit) : ℝ) : WithBot ℝ) := by
  change sum (fun i : Fin B => (xs (i,col,PUnit.unit) : WithBot ℝ)) = _
  exact sum_coe _

#axiomsClean maximum_finite
#axiomsClean exp_sub
#axiomsClean sum_inner
#axiomsClean sum_noninner
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Ideal
