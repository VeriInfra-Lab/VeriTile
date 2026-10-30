import bench.repairs.tritonbench_g.fast_layernorm.IO
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelReference

namespace VeriTile.Bench.Repairs.FastLayernorm.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α : Type} [Inhabited α]

/-- Independent ordered scalar expressions, with logical channel inputs only. -/
def forwardGraph (M : Algebra α) (N B : Nat) (eps : ℝ) (bias : Bool) (v : ForwardValues α B) :=
  forwardResult
    (fun j => (KernelReference.output M N eps bias (forwardX v) (forwardW v) (forwardB v) j).read .real)
    ((KernelReference.savedInverse M N eps (forwardX v)).read .real)
    ((KernelReference.savedMean M N (forwardX v)).read .real)
def backwardGraph (M : Algebra α) (N B : Nat) (v : BackwardValues α B) :=
  backwardResult (fun j => (KernelReference.gradient M N (backwardX v) (backwardDY v)
    (backwardW v) (backwardMean v) (backwardInverse v) j).read .real)
def affineGraph (M : Algebra α) (rows N pid : Nat) (v : AffineValues α rows) :=
  let carry := fun j : Fin 128 => KernelReference.affineCarry M (decide (pid*128+j.val<N))
    (extend (fun row => affineX v row j)) (extend (fun row => affineDY v row j))
    (extend (affineMean v)) (extend (affineInverse v)) rows
  affineResult (fun j => KernelReference.cast M (carry j).1) (fun j => KernelReference.cast M (carry j).2)

namespace Congruence
open KernelReference

theorem input (M : Algebra α) (N : Nat) (x y : Fin B → α)
    (h : ∀ j,j.val<N → x j=y j) : KernelReference.input M N x=KernelReference.input M N y := by
  funext j
  by_cases active : j.val<N <;> simp [KernelReference.input,active,h j]

theorem mean (M : Algebra α) (N : Nat) (x y : Fin B → α)
    (h : ∀ j,j.val<N → x j=y j) : KernelReference.mean M N x=KernelReference.mean M N y := by
  unfold KernelReference.mean
  rw [input M N x y h]

theorem centered (M : Algebra α) (N : Nat) (x y : Fin B → α)
    (h : ∀ j,j.val<N → x j=y j) : KernelReference.centered M N x=KernelReference.centered M N y := by
  funext j
  simp only [KernelReference.centered,input M N x y h,mean M N x y h]

theorem inverse (M : Algebra α) (N : Nat) (eps : ℝ) (x y : Fin B → α)
    (h : ∀ j,j.val<N → x j=y j) : KernelReference.inverse M N eps x=KernelReference.inverse M N eps y := by
  unfold KernelReference.inverse
  rw [centered M N x y h]

theorem output (M : Algebra α) (N : Nat) (eps : ℝ) (bias : Bool) (x w b x' w' b' : Fin B → α)
    (hx : ∀ j,j.val<N → x j=x' j) (hw : ∀ j,j.val<N → w j=w' j)
    (hb : bias=Bool.true → ∀ j,j.val<N → b j=b' j) :
    KernelReference.output M N eps bias x w b=KernelReference.output M N eps bias x' w' b' := by
  funext j
  unfold KernelReference.output
  rw [centered M N x x' hx,inverse M N eps x x' hx,input M N w w' hw]
  cases bias
  · rfl
  · rw [input M N b b' (hb rfl)]

theorem gradient (M : Algebra α) (N : Nat) (x dy w x' dy' w' : Fin B → α) (avg inv : α)
    (hx : ∀ j,j.val<N → x j=x' j) (hdy : ∀ j,j.val<N → dy j=dy' j)
    (hw : ∀ j,j.val<N → w j=w' j) :
    KernelReference.gradient M N x dy w avg inv=KernelReference.gradient M N x' dy' w' avg inv := by
  funext j
  simp only [KernelReference.gradient,KernelReference.normalized,KernelReference.weighted,
    KernelReference.product,KernelReference.reduce,input M N x x' hx,input M N dy dy' hdy,input M N w w' hw]

theorem affine (M : Algebra α) (active : Bool) (x dy avg inv x' dy' avg' inv' : Nat → α)
    (rows : Nat) (hx : active=Bool.true → ∀ row,row<rows → x row=x' row)
    (hdy : active=Bool.true → ∀ row,row<rows → dy row=dy' row)
    (havg : ∀ row,row<rows → avg row=avg' row) (hinv : ∀ row,row<rows → inv row=inv' row) :
    KernelReference.affineCarry M active x dy avg inv rows=
      KernelReference.affineCarry M active x' dy' avg' inv' rows := by
  induction rows with
  | zero => rfl
  | succ rows ih =>
    rw [KernelReference.affineCarry,KernelReference.affineCarry,
      ih (fun h row hr => hx h row (by omega)) (fun h row hr => hdy h row (by omega))
        (fun row hr => havg row (by omega)) (fun row hr => hinv row (by omega))]
    cases active
    · rfl
    · simp [KernelReference.affineTerm,hx rfl rows (by omega),hdy rfl rows (by omega),
        havg rows (by omega),hinv rows (by omega)]
end Congruence
end
end VeriTile.Bench.Repairs.FastLayernorm.Interface
