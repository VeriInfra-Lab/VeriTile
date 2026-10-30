import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.ScalarAlgebra

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open scoped BigOperators

def realOps : Ops ℝ where
  zero := 0
  integer := Int.cast
  cast := id
  add := (·+·)
  sub := (·-·)
  mul := (·*·)
  fma := fun x y z => x*y+z
  exp2 := fun x => Real.exp (x*Real.log 2)
  log2 := fun x => Real.log x/Real.log 2

namespace RealScalar
@[simp] theorem real_zero : realOps.zero=(0:ℝ) := rfl
@[simp] theorem real_cast (x:ℝ) : realOps.cast x=x := rfl
@[simp] theorem real_mul (x y:ℝ) : realOps.mul x y=x*y := rfl
@[simp] theorem real_add (x y:ℝ) : realOps.add x y=x+y := rfl
@[simp] theorem real_fma (x y z:ℝ) : realOps.fma x y z=x*y+z := rfl


theorem rate_positive (c : Shape) (h : Nat) : 0<MathTarget.headRate c h := by
  have hlog : (0:ℝ)<Real.log 2 := Real.log_pos (by norm_num)
  have hindex : (0:ℝ)≤(h%c.H:Nat) := Nat.cast_nonneg _
  have he : (-5-(h%c.H:Nat):ℝ)*Real.log 2<0 :=
    mul_neg_of_neg_of_pos (by linarith) hlog
  exact sub_pos.mpr (by simpa [MathTarget.headRate] using Real.exp_lt_one_iff.mpr he)

theorem log_rate (c : Shape) (h : Nat) :
    logRate realOps c h=Real.log (MathTarget.headRate c h)/Real.log 2 := by
  simp only [logRate,realOps,MathTarget.headRate]
  norm_cast
  simp only [Int.cast_sub,Int.cast_natCast]

theorem decay_nat (c : Shape) (h n : Nat) :
    decay realOps c h n=(MathTarget.headRate c h)^n := by
  have hn : Real.log (2:ℝ)≠0 := ne_of_gt (Real.log_pos (by norm_num))
  change Real.exp (((n:ℝ)*logRate realOps c h)*Real.log 2)=_
  rw [log_rate]
  rw [mul_assoc,div_mul_cancel₀ _ hn,Real.exp_nat_mul,Real.exp_log (rate_positive c h)]

theorem chain_sum (n : Nat) (l r : Nat→ℝ) (seed : ℝ) :
    chain realOps n l r seed=seed+∑ i∈Finset.range n,l i*r i := by
  induction n with
  | zero => simp [chain]
  | succ n ih => simp only [chain,real_fma,ih,Finset.sum_range_succ]; ring

theorem block_sum (width blocks : Nat) (l r : Nat→ℝ) (seed : ℝ) :
    blockChain realOps width l r blocks seed=seed+∑ i∈Finset.range (blocks*width),l i*r i := by
  induction blocks with
  | zero => simp [blockChain]
  | succ n ih =>
    simp only [blockChain,chain_sum,ih,Nat.succ_mul,Finset.sum_range_add]
    ring

theorem masked_key (c : Shape) (row : Nat) (hr : row<c.T)
    (xs : Inputs ℝ) (h e : Nat) :
    qAt realOps c xs h row e=(if e<c.K then xs.q h row e else 0) ∧
    kAt realOps c xs h row e=(if e<c.K then xs.k h row e else 0) := by
  simp [qAt,kAt,masked,hr,realOps]

theorem time_lane (c : Shape) (chunk t : Nat) (hc : chunk<c.chunks) (ht : t<64) :
    chunk*64+t<c.T ↔ t<c.length chunk := by
  have hs:=c.chunk_start chunk hc
  dsimp [Shape.length]
  omega

theorem nextH_sum (c : Shape) (xs : Inputs ℝ) (h chunk e j : Nat) (carry : Tile ℝ)
    (hc : chunk<c.chunks) (he : e<c.K) (hj : j<c.V) :
    Body.nextH realOps c xs h chunk carry e j=
      (MathTarget.headRate c h)^(c.length chunk)*carry e j+
      ∑ t∈Finset.range (c.length chunk),xs.k h (chunk*64+t) e*
        (xs.v h (chunk*64+t) j*(MathTarget.headRate c h)^(c.length chunk-1-t)) := by
  simp only [Body.nextH,chain_sum,decay_nat,real_mul,real_cast]
  congr 1
  rw [MathTarget.sum_prefix _ _ 64 (c.length_le chunk)]
  apply Finset.sum_congr rfl
  intro t ht
  have ht64:=Finset.mem_range.mp ht
  have hp:=time_lane c chunk t hc ht64
  by_cases hl : t<c.length chunk
  · have heq : (c.length chunk:Int)-t-1=((c.length chunk-1-t:Nat):Int) := by omega
    simp [kAt,vAt,masked,he,hj,hp.mpr hl,hl,heq,decay_nat]
  · have hn : ¬chunk*64+t<c.T := fun H=>hl (hp.mp H)
    simp [kAt,vAt,masked,hn,hl,realOps]

theorem nextDH_sum (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h chunk e j : Nat)
    (carry : Tile ℝ) (hc : chunk<c.chunks) (he : e<c.K) (hj : j<c.V) :
    Body.nextDH realOps c xs scale h chunk carry e j=
      (MathTarget.headRate c h)^(c.length chunk)*carry e j+
      ∑ t∈Finset.range (c.length chunk),(xs.q h (chunk*64+t) e*scale)*
        (xs.incoming h (chunk*64+t) j*(MathTarget.headRate c h)^(t+1)) := by
  simp only [Body.nextDH,chain_sum,decay_nat,real_mul,real_cast]
  congr 1
  rw [MathTarget.sum_prefix _ _ 64 (c.length_le chunk)]
  apply Finset.sum_congr rfl
  intro t ht
  have hp:=time_lane c chunk t hc (Finset.mem_range.mp ht)
  by_cases hl : t<c.length chunk
  · have heq : (t:Int)+1=((t+1:Nat):Int) := by omega
    simp only [qAt,gAt,masked,if_pos (And.intro (hp.mpr hl) he),
      if_pos (And.intro (hp.mpr hl) hj),if_pos hl,heq,decay_nat]
    ring
  · have hn : ¬chunk*64+t<c.T := fun H=>hl (hp.mp H)
    simp [qAt,gAt,masked,hn,hl,realOps]

theorem history_scalar (c : Shape) (xs : Inputs ℝ) (h n e j : Nat)
    (hn : n≤c.chunks) (he : e<c.K) (hj : j<c.V) :
    Reference.history realOps c xs h n e j=ScalarSource.forwardCarry c xs Bool.true h n e j := by
  induction n with
  | zero => simp [Reference.history,ScalarSource.forwardCarry,masked,he,hj,realOps]
  | succ n ih =>
    rw [Reference.history,nextH_sum c xs h n e j _ (by omega) he hj,
      ih (by omega),ScalarSource.forwardCarry]

theorem reverse_scalar (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h n e j : Nat)
    (hn : n≤c.chunks) (he : e<c.K) (hj : j<c.V) :
    Reference.reverseCarry realOps c xs scale h n e j=ScalarSource.reverseCarry c xs scale h n e j := by
  induction n with
  | zero => simp [Reference.reverseCarry,ScalarSource.reverseCarry,masked,he,hj,realOps]
  | succ n ih =>
    rw [Reference.reverseCarry,nextDH_sum c xs scale h (c.chunks-1-n) e j _ (by omega) he hj,
      ih (by omega),ScalarSource.reverseCarry]

#axiomsClean decay_nat
#axiomsClean nextH_sum
#axiomsClean nextDH_sum
#axiomsClean history_scalar
#axiomsClean reverse_scalar
end RealScalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
