import bench.tritonbench_g.rope_backward_transform.Proofs.TypedFrame
import Mathlib.Data.Nat.Log
import Mathlib.Tactic.Ring

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
noncomputable section

structure Request where
  (batches sequence qHeads kHeads dimension : Nat)
  (cosRows sinRows cosRowStride cosColStride sinRowStride sinColStride : Nat)
def Request.rows (r : Request) := r.batches*r.sequence
def Request.half (r : Request) := r.dimension/2
def Request.heads (r : Request) (key : Bool) := if key then r.kHeads else r.qHeads
def pad (n : Nat) := 2^Nat.clog 2 (max n 1)
def Request.paddedHeads (r : Request) (key : Bool) := pad (r.heads key)
def Request.paddedDim (r : Request) := pad r.dimension
def Request.size (r : Request) (key : Bool) := r.rows*r.heads key*r.dimension
def Request.offset (r : Request) (key : Bool) (row head col : Nat) :=
  (row*r.heads key+head)*r.dimension+col
def Request.external (r : Request) (key : Bool) (b head time col : Nat) :=
  ((b*r.heads key+head)*r.sequence+time)*r.dimension+col
def region (key : Bool) : RegionName := if key then "K" else "Q"
def parameters (r : Request) : Source.Parameters where
  q := "Q"
  k := "K"
  cos := "C"
  sin := "S"
  qStride := r.qHeads*r.dimension
  kStride := r.kHeads*r.dimension
  cosStride := r.cosRowStride
  sinStride := r.sinRowStride
  cosColStride := r.cosColStride
  sinColStride := r.sinColStride
  length := r.sequence
  qHeads := r.qHeads
  kHeads := r.kHeads
  headDim := r.dimension
  paddedQHeads := r.paddedHeads Bool.false
  paddedKHeads := r.paddedHeads Bool.true
  paddedHeadDim := r.paddedDim
  backward := Bool.true

namespace Spec
/-- Fixed-coefficient transpose of the forward rotation; no unit-circle premise. -/
def pair (a b c s : ℝ) : ℝ×ℝ := (a*c+b*s,b*c-a*s)
def forward (a b c s : ℝ) : ℝ×ℝ := (a*c-b*s,b*c+a*s)
theorem adjoint (a b u v c s : ℝ) :
    (forward a b c s).1*u+(forward a b c s).2*v=
      a*(pair u v c s).1+b*(pair u v c s).2 := by simp only [forward,pair]; ring
/-- Independent logical target on original [B,H,S,D] tensors and [L,D/2] coefficients. -/
def result (r : Request) (input : Bool→Nat→Nat→Nat→Nat→ℝ)
    (cosine sine : Nat→Nat→ℝ) (key : Bool) (b h time col : Nat) : ℝ :=
  if col<2*r.half then
    let lane := if col<r.half then col else col-r.half
    let value := pair (input key b h time lane) (input key b h time (lane+r.half))
      (cosine time lane) (sine time lane)
    if col<r.half then value.1 else value.2
  else input key b h time col
end Spec

structure Operations (α : Type) where
  zero : α
  cast : α→α
  add : α→α→α
  sub : α→α→α
  mul : α→α→α
def Operations.backend (op : Operations α) : Algebra α where
  literal := fun _ _ _ => op.zero
  negInf := op.zero
  binary := fun _ _ code => match code with
    | .add => op.add | .sub => op.sub | .mul => op.mul | _ => fun a _ => a
  unary := fun _ _ => id
  cast := fun _ _ _ => op.cast
  fromNat := fun _ _ => op.zero
  fromInt := fun _ _ => op.zero
  fp32Bits := fun _ => op.zero
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => op.zero
  reduceSum := fun _ {_} _ _ _ _ => op.zero
def scalarPair (op : Operations α) (second : Bool) (a b c s : α) : α :=
  let x := if second then op.cast b else op.cast a
  let y := if second then op.cast a else op.cast b
  op.cast ((if second then op.sub else op.add) (op.mul x c) (op.mul y s))
def ideal : Operations ℝ := ⟨0,id,(·+·),(·-·),(·*·)⟩
theorem ideal_pair (second : Bool) (a b c s : ℝ) :
    scalarPair ideal second a b c s=if second then (Spec.pair a b c s).2 else (Spec.pair a b c s).1 := by
  cases second <;> rfl

theorem pad_covers (n : Nat) : n≤pad n :=
  le_trans (Nat.le_max_left ..) (Nat.le_pow_clog (by decide) _)
theorem half_covers (r : Request) : r.half≤r.paddedDim/2 :=
  Nat.div_le_div_right (pad_covers r.dimension)
theorem padded_half_le (n : Nat) : pad n/2≤n := by
  by_cases small : n≤1
  · interval_cases n <;> norm_num [pad,Nat.clog_one_right]
  · have hn : 1<n := by omega
    have hp := Nat.pow_pred_clog_lt_self (b:=2) (by decide) hn
    simp only [Nat.pred_eq_sub_one] at hp
    have hz := Nat.clog_pos (b:=2) (by decide) hn
    have he : Nat.clog 2 n=(Nat.clog 2 n-1)+1 := by omega
    unfold pad
    rw [max_eq_left (by omega),he,pow_succ,Nat.mul_div_left _ (by decide)]
    omega
theorem row_bound (r : Request) (b t : Nat) (hb : b<r.batches) (ht : t<r.sequence) : b*r.sequence+t<r.rows := by
  have h := Nat.mul_le_mul_right r.sequence (show b+1≤r.batches by omega)
  simp only [Nat.add_mul,Nat.one_mul] at h
  unfold Request.rows
  omega
theorem offset_bound (r : Request) (key : Bool) (row head col : Nat)
    (hr : row<r.rows) (hh : head<r.heads key) (hc : col<r.dimension) :
    r.offset key row head col<r.size key := by
  have h1 := Nat.mul_le_mul_right (r.heads key) (show row+1≤r.rows by omega)
  have h2 := Nat.mul_le_mul_right r.dimension (show row*r.heads key+head+1≤r.rows*r.heads key by nlinarith)
  unfold Request.offset Request.size
  nlinarith
theorem offset_injective (r : Request) (key : Bool) {a h c a' h' c' : Nat}
    (hh : h<r.heads key) (hc : c<r.dimension) (hh' : h'<r.heads key) (hc' : c'<r.dimension)
    (eq : r.offset key a h c=r.offset key a' h' c') : a=a' ∧ h=h' ∧ c=c' := by
  have hd : 0<r.dimension := by omega
  have hp : 0<r.heads key := by omega
  have cc := congrArg (fun n => n%r.dimension) eq
  simp [Request.offset,Nat.add_mod,Nat.mod_eq_of_lt hc,Nat.mod_eq_of_lt hc'] at cc
  have div (k x : Nat) (hx : x<r.dimension) : (k*r.dimension+x)/r.dimension=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hd,Nat.div_eq_of_lt hx]; simp
  have ah := congrArg (fun n => n/r.dimension) eq
  simp only [Request.offset,div _ _ hc,div _ _ hc'] at ah
  have hh1 := congrArg (fun n => n%(r.heads key)) ah
  simp [Nat.add_mod,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh'] at hh1
  have divh (k x : Nat) (hx : x<r.heads key) : (k*r.heads key+x)/r.heads key=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hx]; simp
  have aa := congrArg (fun n => n/(r.heads key)) ah
  simp only [divh _ _ hh,divh _ _ hh'] at aa
  exact ⟨aa,hh1,cc⟩

#axiomsClean Spec.adjoint
#axiomsClean offset_injective
#specNonCircular Spec.result avoiding [Source.kernel,Source.referenceMemory,scalarPair,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
