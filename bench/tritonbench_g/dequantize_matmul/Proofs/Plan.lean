import VeriTile.Triton
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 2000000

/-- Fixed K8 instruction families, not whole dot operations. -/
inductive MMAKind where
  | mmaSyncM16N8K8RowColF32TF32TF32F32
  | wgmmaM64N8K8F32TF32TF32
  deriving DecidableEq

inductive TF32Mode where
  | nearestEven | nearestAway
  deriving DecidableEq

structure Arithmetic (α : Type) where
  zero : α
  signed8 : BitVec 8 → α
  load : α → α
  store : α → α
  mul : α → α → α
  add : α → α → α
  fma : α → α → α → α
  tf32 : TF32Mode → α → α
  mma8 : MMAKind → (Fin 8 → α) → (Fin 8 → α) → α → α

/-- A selected implementation's finite contraction tree. All source operands,
seeds, partial sums and optional hardware-padding lanes are explicit. -/
inductive Plan (K : Nat) where
  | zero
  | product (k : Fin K)
  | fma (k : Fin K) (seed : Plan K)
  | add (left right : Plan K)
  | mma8 (kind : MMAKind) (conversion : TF32Mode) (lanes : Fin 8 → Option (Fin K)) (seed : Plan K)

def Plan.terms : Plan K → List (Fin K)
  | .zero => []
  | .product k => [k]
  | .fma k seed => k :: seed.terms
  | .add left right => left.terms ++ right.terms
  | .mma8 _ _ lanes seed => (List.ofFn lanes).filterMap id ++ seed.terms

def Plan.Complete (p : Plan K) : Prop := p.terms.Perm (List.finRange K)

def Plan.eval (a : Arithmetic α) (xs ys : Fin K → α) : Plan K → α
  | .zero => a.zero
  | .product k => a.mul (a.load (xs k)) (a.load (ys k))
  | .fma k seed => a.fma (a.load (xs k)) (a.load (ys k)) (seed.eval a xs ys)
  | .add left right => a.add (left.eval a xs ys) (right.eval a xs ys)
  | .mma8 kind conversion lanes seed => a.mma8 kind
      (fun i => (lanes i).map (fun k => a.tf32 conversion (a.load (xs k))) |>.getD (a.tf32 conversion a.zero))
      (fun i => (lanes i).map (fun k => a.tf32 conversion (a.load (ys k))) |>.getD (a.tf32 conversion a.zero))
      (seed.eval a xs ys)

def realArithmetic : Arithmetic ℝ where
  zero := 0
  signed8 := fun q => q.toInt
  load := id
  store := id
  mul := (·*·)
  add := (·+·)
  fma := fun a b z => a*b+z
  tf32 := fun _ => id
  mma8 := fun _ a b z => (List.ofFn (fun i => a i*b i)).sum+z

theorem option_sum (xs : List (Option β)) (f : β → ℝ) :
    (xs.map (fun x => (x.map f).getD 0)).sum =
      ((xs.filterMap id).map f).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih => cases x <;> simp [ih]

theorem Plan.eval_real (p : Plan K) (a b : Fin K → ℝ) :
    p.eval realArithmetic a b = (p.terms.map (fun k => a k*b k)).sum := by
  induction p with
  | zero => rfl
  | product k => simp [eval,terms,realArithmetic]
  | fma k seed ih =>
    change a k*b k + seed.eval realArithmetic a b = _
    rw [ih]
    rfl
  | add left right il ir =>
    change left.eval realArithmetic a b + right.eval realArithmetic a b = _
    rw [il,ir]
    simp only [terms,List.map_append,List.sum_append]
  | mma8 kind conversion lanes seed ih =>
    change (List.ofFn (fun i => ((lanes i).map a).getD 0 * ((lanes i).map b).getD 0)).sum +
      seed.eval realArithmetic a b = _
    rw [ih]
    simp only [terms,List.map_append,List.sum_append]
    congr 1
    have per (i : Fin 8) :
        ((lanes i).map a).getD 0 * ((lanes i).map b).getD 0 =
          ((lanes i).map (fun k => a k*b k)).getD 0 := by cases lanes i <;> simp
    simp_rw [per]
    simpa only [List.map_ofFn,Function.comp_def] using
      option_sum (List.ofFn lanes) (fun k => a k*b k)

/-- All K terms, with no assumed whole-reduction numerical equation. -/
theorem Plan.real_complete (p : Plan K) (complete : p.Complete) (a b : Fin K → ℝ) :
    p.eval realArithmetic a b = (List.ofFn (fun k => a k*b k)).sum := by
  rw [p.eval_real]
  have h := List.Perm.sum_eq (List.Perm.map (fun k => a k*b k) complete)
  simpa only [List.finRange,List.map_ofFn,Function.comp_def,id_eq] using h

structure LocalCorrespondence (actual reference : Arithmetic α) : Prop where
  zero : actual.zero=reference.zero
  signed8 : ∀ q,actual.signed8 q=reference.signed8 q
  load : ∀ x,actual.load x=reference.load x
  store : ∀ x,actual.store x=reference.store x
  mul : ∀ x y,actual.mul x y=reference.mul x y
  add : ∀ x y,actual.add x y=reference.add x y
  fma : ∀ x y z,actual.fma x y z=reference.fma x y z
  tf32 : ∀ mode x,actual.tf32 mode x=reference.tf32 mode x
  mma8 : ∀ kind xs ys z,actual.mma8 kind xs ys z=reference.mma8 kind xs ys z

theorem LocalCorrespondence.arithmetic_eq (h : LocalCorrespondence a b) : a=b := by
  cases a
  cases b
  obtain ⟨hz,hq,hl,hs,hm,ha,hf,ht,hi⟩ := h
  simp only [Arithmetic.mk.injEq]
  exact ⟨hz,funext hq,funext hl,funext hs,funext fun x => funext (hm x),
    funext fun x => funext (ha x),funext fun x => funext fun y => funext (hf x y),(funext fun mode => funext (ht mode)),
    funext fun kind => funext fun xs => funext fun ys => funext (hi kind xs ys)⟩

theorem Plan.local_to_complete (p : Plan K) (actual reference : Arithmetic α)
    (h : LocalCorrespondence actual reference)
    (xs ys : Fin K → α) : p.eval actual xs ys=p.eval reference xs ys := by
  rw [h.arithmetic_eq]

#axiomsClean Plan.eval_real
#axiomsClean Plan.real_complete
#axiomsClean Plan.local_to_complete
end
end VeriTile.Bench.Repairs.DequantizeMatmul
