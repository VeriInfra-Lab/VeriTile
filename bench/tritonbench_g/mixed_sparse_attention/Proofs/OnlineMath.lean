import bench.tritonbench_g.mixed_sparse_attention.Proofs.RealBackend
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online
open VeriTile Triton
noncomputable section

structure Tile (N : Nat) where
  active : Fin N → Bool
  score : Fin N → ℝ
  value : Fin N → ℝ

structure Acc where
  maximum : WithBot ℝ
  denominator : ℝ
  numerator : ℝ

def denominator {N : Nat} (t : Tile N) : ℝ := ∑ j,if t.active j then pow2 (t.score j) else 0
def numerator {N : Nat} (t : Tile N) : ℝ := ∑ j,if t.active j then pow2 (t.score j)*t.value j else 0
def nextMaximum {N : Nat} (a : Acc) (t : Tile N) : WithBot ℝ :=
  max a.maximum (Finset.univ.sup fun j=>if t.active j then (t.score j : WithBot ℝ) else ⊥)
def alpha (old : WithBot ℝ) (shift : ℝ) : ℝ :=
  match old with | none=>0 | some m=>pow2 (m-shift)
def weight {N : Nat} (t : Tile N) (shift : ℝ) (j : Fin N) : ℝ :=
  if t.active j then pow2 (t.score j-shift) else 0

/-- Exact Real form of the repaired finite-temporary-shift update. The
stored maximum remains bottom when no occurrence has yet been selected. -/
def step {N : Nat} (a : Acc) (t : Tile N) : Acc :=
  let maximum:=nextMaximum a t
  let shift:=maximum.unbotD 0
  let rescale:=alpha a.maximum shift
  ⟨maximum,a.denominator*rescale+∑ j,weight t shift j,
    a.numerator*(a.denominator*0+rescale)+∑ j,weight t shift j*t.value j⟩

def Consistent (a : Acc) (L T : ℝ) : Prop :=
  a.denominator=pow2 (-a.maximum.unbotD 0)*L ∧
  a.numerator=pow2 (-a.maximum.unbotD 0)*T ∧
  (a.maximum=⊥ → L=0 ∧ T=0)

theorem weighted_factor {N : Nat} (t : Tile N) (shift : ℝ) (v : Fin N → ℝ) :
    (∑ j,weight t shift j*v j)=pow2 (-shift)*(∑ j,if t.active j then pow2 (t.score j)*v j else 0) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  cases hj : t.active j
  · simp [weight,hj]
  · have hpow : pow2 (t.score j-shift)=pow2 (-shift)*pow2 (t.score j) := by
      rw [←pow2_add]
      congr 1
      ring
    simp [weight,hj,hpow,mul_assoc]

theorem weight_factor {N : Nat} (t : Tile N) (shift : ℝ) :
    (∑ j,weight t shift j)=pow2 (-shift)*denominator t := by
  simpa [denominator] using weighted_factor t shift (fun _=>1)

theorem rescale_factor (old : WithBot ℝ) (shift total : ℝ) (h : old=⊥ → total=0) :
    pow2 (-old.unbotD 0)*total*alpha old shift=pow2 (-shift)*total := by
  cases old with
  | bot=>simp [h rfl,alpha]
  | coe x=>
    simp only [WithBot.unbotD_coe,alpha]
    have hp : pow2 (-x)*pow2 (x-shift)=pow2 (-shift) := by
      rw [←pow2_add]
      congr 1
      ring
    rw [mul_right_comm,hp]

theorem empty_maximum {N : Nat} (a : Acc) (t : Tile N) (h : nextMaximum a t=⊥) :
    a.maximum=⊥ ∧ ∀ j,t.active j=Bool.false := by
  have ho : a.maximum≤(⊥ : WithBot ℝ) := by
    rw [←h]
    exact le_max_left _ _
  refine ⟨le_bot_iff.mp ho,?_⟩
  intro j
  cases hj : t.active j
  · rfl
  · have he : (t.score j : WithBot ℝ)≤nextMaximum a t := by
      apply le_trans ?_ (le_max_right _ _)
      have hle:=Finset.le_sup (f:=fun k : Fin N=>if t.active k then (t.score k : WithBot ℝ) else ⊥) (Finset.mem_univ j)
      simpa [hj] using hle
    rw [h] at he
    simp at he

theorem step_consistent {N : Nat} (a : Acc) (t : Tile N) (L T : ℝ) (h : Consistent a L T) :
    Consistent (step a t) (L+denominator t) (T+numerator t) := by
  obtain ⟨hl,ht,he⟩:=h
  refine ⟨?_,?_,?_⟩
  · change a.denominator*alpha a.maximum _+_=pow2 _*_
    rw [hl,rescale_factor _ _ _ (fun hx=>(he hx).1),weight_factor]
    simp only [step]
    ring
  · change a.numerator*(a.denominator*0+alpha a.maximum _)+_=pow2 _*_
    simp only [mul_zero,zero_add]
    rw [ht,rescale_factor _ _ _ (fun hx=>(he hx).2),weighted_factor]
    simp only [step,numerator]
    ring
  · intro hn
    obtain ⟨ho,hm⟩:=empty_maximum a t hn
    obtain ⟨hL,hT⟩:=he ho
    simp [hL,hT,denominator,numerator,hm]

theorem initial_consistent : Consistent ⟨⊥,0,0⟩ 0 0 := by simp [Consistent]

/-- Empty tiles and empty prefixes are included in this invariant. -/
theorem fold_consistent {N : Nat} (tiles : List (Tile N)) (a : Acc) (L T : ℝ) (h : Consistent a L T) :
    Consistent (tiles.foldl step a)
      (L+(tiles.map denominator).sum) (T+(tiles.map numerator).sum) := by
  induction tiles generalizing a L T with
  | nil=>simpa using h
  | cons t ts ih=>
    simpa [List.foldl_cons,List.map_cons,List.sum_cons,add_assoc] using
      ih (step a t) (L+denominator t) (T+numerator t) (step_consistent a t L T h)

theorem result {N : Nat} (tiles : List (Tile N)) :
    let final:=tiles.foldl step ⟨⊥,0,0⟩
    final.numerator/final.denominator=(tiles.map numerator).sum/(tiles.map denominator).sum := by
  obtain ⟨hl,ht,_⟩:=fold_consistent tiles ⟨⊥,0,0⟩ 0 0 initial_consistent
  simp only [zero_add] at hl ht
  dsimp only
  rw [hl,ht,mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))]

#axiomsClean step_consistent
#axiomsClean fold_consistent
#axiomsClean result
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online
