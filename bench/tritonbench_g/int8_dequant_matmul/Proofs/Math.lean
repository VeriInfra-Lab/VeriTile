import bench.tritonbench_g.int_scaled_matmul.Proofs.Geometry
import bench.tritonbench_g.dequantize_matmul.Proofs.FPReference

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

structure Shape where
  M : Nat
  N : Nat
  K : Nat

structure Config where
  BM : Nat
  BN : Nat
  BK : Nat
  S : Nat
  positiveM : 0 < BM
  positiveN : 0 < BN
  positiveK : 0 < BK
  positiveS : 0 < S

def steps (s : Shape) (c : Config) := (s.K+c.BK*c.S-1)/(c.BK*c.S)
def sourceK (c : Config) (t z l : Nat) := t*(c.BK*c.S)+z*c.BK+l
def geometry (s : Shape) (c : Config) : IntScaledMatmul.Revision001.Geometry.Layout :=
  ⟨s.M,s.N,c.BM,c.BN,8,s.N,1⟩
abbrev MatrixCell (s : Shape) := Fin s.M × Fin s.N

def rsum (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => rsum f n+f n

@[simp] theorem rsum_zero (n : Nat) : rsum (fun _ => 0) n=0 := by
  induction n with | zero => rfl | succ n ih => simp [rsum,ih]
theorem rsum_congr (f g : Nat → ℝ) (n : Nat) (h : ∀ i,i<n → f i=g i) : rsum f n=rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [rsum,ih (fun i hi => h i (by omega)),h n (by omega)]
theorem rsum_add (f g : Nat → ℝ) (n : Nat) :
    rsum (fun i => f i+g i) n=rsum f n+rsum g n := by
  induction n with | zero => simp [rsum] | succ n ih => simp only [rsum,ih]; ring
theorem rsum_scale (a : ℝ) (f : Nat → ℝ) (n : Nat) :
    rsum (fun i => a*f i) n=a*rsum f n := by
  induction n with | zero => simp [rsum] | succ n ih => simp only [rsum,ih]; ring
theorem rsum_swap (f : Nat → Nat → ℝ) (m n : Nat) :
    rsum (fun i => rsum (f i) n) m=rsum (fun j => rsum (fun i => f i j) m) n := by
  induction m with
  | zero => simp [rsum]
  | succ m ih => simp only [rsum,ih,rsum_add]
theorem rsum_append (f : Nat → ℝ) (a b : Nat) :
    rsum f (a+b)=rsum f a+rsum (fun i => f (a+i)) b := by
  induction b with | zero => simp [rsum] | succ b ih => simp [rsum,ih,Nat.add_assoc,add_assoc]
theorem rsum_blocks (f : Nat → ℝ) (B n : Nat) :
    rsum (fun t => rsum (fun l => f (t*B+l)) B) n=rsum f (n*B) := by
  induction n with
  | zero => simp [rsum]
  | succ n ih => simp only [rsum,ih,Nat.succ_mul,rsum_append]
theorem rsum_padding (f : Nat → ℝ) (K n : Nat) :
    rsum (fun i => if i<K then f i else 0) n=rsum f (min K n) := by
  induction n with
  | zero => simp [rsum]
  | succ n ih =>
    by_cases h : n<K
    · simp only [rsum,ih,if_pos h,Nat.min_eq_right (by omega : n≤K),Nat.min_eq_right (by omega : n+1≤K)]
    · simp only [rsum,ih,if_neg h,Nat.min_eq_left (by omega : K≤n),Nat.min_eq_left (by omega : K≤n+1),add_zero]

/-- The exact triple index used by both source K branches, including all split
IDs, is a disjoint partition of every active K term. No arithmetic plan or
native output assumption occurs in this theorem. -/
theorem split_partition (s : Shape) (c : Config) (f : Nat → ℝ) :
    rsum (fun z => rsum (fun t => rsum (fun l =>
      if sourceK c t z l<s.K then f (sourceK c t z l) else 0) c.BK) (steps s c)) c.S=
      rsum f s.K := by
  rw [rsum_swap]
  have inner (t : Nat) :
      rsum (fun z => rsum (fun l => if sourceK c t z l<s.K then f (sourceK c t z l) else 0) c.BK) c.S=
      rsum (fun q => if t*(c.BK*c.S)+q<s.K then f (t*(c.BK*c.S)+q) else 0) (c.BK*c.S) := by
    simpa only [sourceK,Nat.add_assoc,Nat.mul_comm c.S c.BK] using
      rsum_blocks (fun q => if t*(c.BK*c.S)+q<s.K then f (t*(c.BK*c.S)+q) else 0) c.BK c.S
  simp_rw [inner]
  rw [rsum_blocks (fun i => if i<s.K then f i else 0),rsum_padding,Nat.min_eq_left]
  exact IntScaledMatmul.Revision001.LogicalSum.ceil_cover _ _ (Nat.mul_pos c.positiveK c.positiveS)

theorem sourceK_unique (c : Config) (t z l t' z' l' : Nat)
    (hz : z<c.S) (hz' : z'<c.S) (hl : l<c.BK) (hl' : l'<c.BK)
    (same : sourceK c t z l=sourceK c t' z' l') : t=t' ∧ z=z' ∧ l=l' := by
  have bounds : z*c.BK+l<c.BK*c.S := by nlinarith
  have bounds' : z'*c.BK+l'<c.BK*c.S := by nlinarith
  have ht : t=t' := by
    have h := congrArg (fun x => x/(c.BK*c.S)) same
    simp only [sourceK,←Nat.add_assoc,Nat.add_assoc] at h
    have e : t*(c.BK*c.S)+(z*c.BK+l)=sourceK c t z l := by simp [sourceK,Nat.add_assoc]
    have e' : t'*(c.BK*c.S)+(z'*c.BK+l')=sourceK c t' z' l' := by simp [sourceK,Nat.add_assoc]
    have d := Nat.div_eq_of_lt bounds
    have d' := Nat.div_eq_of_lt bounds'
    have p := Nat.mul_pos c.positiveK c.positiveS
    rw [Nat.mul_comm t (c.BK*c.S),Nat.mul_comm t' (c.BK*c.S),Nat.mul_add_div p,Nat.mul_add_div p,d,d'] at h
    omega
  subst t'
  have equal : z*c.BK+l=z'*c.BK+l' := by simp only [sourceK] at same; omega
  have h := congrArg (fun x => x/c.BK) equal
  change (z*c.BK+l)/c.BK=(z'*c.BK+l')/c.BK at h
  rw [Nat.mul_comm z c.BK,Nat.mul_comm z' c.BK,Nat.mul_add_div c.positiveK,Nat.mul_add_div c.positiveK,
    Nat.div_eq_of_lt hl,Nat.div_eq_of_lt hl'] at h
  have hz' : z=z' := by omega
  subst z'
  exact ⟨rfl,rfl,by omega⟩

/-- K-tail masking and EVEN_K's unmasked loads have the same active terms. -/
theorem source_mask (s : Shape) (c : Config) (t z l : Nat) (ht : t<steps s c)
    (hz : z<c.S) (hl : l<c.BK) :
    (z*c.BK+l<s.K-t*(c.BK*c.S)) ↔ sourceK c t z l<s.K := by
  simp only [sourceK]
  omega
theorem even_active (s : Shape) (c : Config) (even : s.K%(c.BK*c.S)=0)
    (t z l : Nat) (ht : t<steps s c) (hz : z<c.S) (hl : l<c.BK) : sourceK c t z l<s.K := by
  have exactK := IntScaledMatmul.Revision001.LogicalSum.ceil_exact s.K (c.BK*c.S)
    (Nat.mul_pos c.positiveK c.positiveS) even
  change s.K=steps s c*(c.BK*c.S) at exactK
  have tail : z*c.BK+l<c.BK*c.S := by nlinarith
  unfold sourceK
  nlinarith

structure RealInput (s : Shape) where
  A : Nat → Nat → ℝ
  B : Nat → Nat → ℝ
  x : Nat → ℝ
  w : Nat → ℝ
  bias : Option (Nat → ℝ)
def biasAt (x : RealInput s) (n : Nat) := (x.bias.map (fun b => b n)).getD 0
def target (s : Shape) (x : RealInput s) (r n : Nat) :=
  x.w n*(x.x r*(rsum (fun k => x.A r k*x.B k n) s.K/16129))+biasAt x n
def realPartial (s : Shape) (c : Config) (x : RealInput s) (r n z : Nat) :=
  x.w n*(x.x r*(rsum (fun t => rsum (fun l =>
    if sourceK c t z l<s.K then x.A r (sourceK c t z l)*x.B (sourceK c t z l) n else 0)
    c.BK) (steps s c)/16129))+(if z=0 then biasAt x n else 0)
theorem rsum_bias (b : ℝ) (n : Nat) (hn : 0<n) : rsum (fun z => if z=0 then b else 0) n=b := by
  cases n with
  | zero => omega
  | succ n =>
    clear hn
    induction n with
    | zero => simp [rsum]
    | succ n ih => change rsum (fun z => if z=0 then b else 0) (n+1)+(if n+1=0 then b else 0)=b; rw [ih,if_neg (by omega),add_zero]
theorem partials_target (s : Shape) (c : Config) (x : RealInput s) (r n : Nat) :
    rsum (realPartial s c x r n) c.S=target s x r n := by
  unfold realPartial target
  simp only [div_eq_mul_inv]
  rw [rsum_add,rsum_bias _ _ c.positiveS]
  have e (z : Nat) : x.w n*(x.x r*(rsum (fun t => rsum (fun l =>
      if sourceK c t z l<s.K then x.A r (sourceK c t z l)*x.B (sourceK c t z l) n else 0) c.BK)
      (steps s c)*(16129:ℝ)⁻¹))=
    (x.w n*x.x r*(16129:ℝ)⁻¹)*rsum (fun t => rsum (fun l =>
      if sourceK c t z l<s.K then x.A r (sourceK c t z l)*x.B (sourceK c t z l) n else 0) c.BK) (steps s c) := by ring
  simp_rw [e]
  rw [rsum_scale,split_partition s c (fun k => x.A r k*x.B k n)]
  ring

#axiomsClean split_partition
#axiomsClean sourceK_unique
#axiomsClean source_mask
#axiomsClean even_active
#axiomsClean partials_target
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
