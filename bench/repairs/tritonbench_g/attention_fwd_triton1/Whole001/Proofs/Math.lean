import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Source
import Mathlib.Algebra.BigOperators.Ring.Finset

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section
open scoped BigOperators

namespace MathTarget
/-- Independent input-only chunked linear attention. No softmax and no
token-causal truncation inside the current chunk. -/
def output (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head row j : Nat) : ℝ :=
  scale * ∑ u∈Finset.range (min ((row/32+1)*32) c.T),
    ∑ e∈Finset.range c.D, xs.q head row e * xs.k head u e * xs.v head u j
def history (c : Shape) (xs : Inputs ℝ) (head chunk i j : Nat) : ℝ :=
  ∑ u∈Finset.range (min (chunk*32) c.T), xs.k head u i * xs.v head u j

/-- Zero extension of the mathematical K/V contraction, used only to prove
the equivalence of a finite padded sum with its truncated prefix. -/
def kv (c : Shape) (xs : Inputs ℝ) (head u i j : Nat) : ℝ :=
  if u<c.T then xs.k head u i*xs.v head u j else 0

theorem sum_guard (T n : Nat) (f : Nat→ℝ) :
    (∑ u∈Finset.range n, if u<T then f u else 0)=∑ u∈Finset.range (min n T), f u := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ,ih]
    by_cases hn : n<T
    · simp [hn,Nat.min_eq_left (Nat.le_of_lt hn),Nat.min_eq_left (show n+1≤T by omega),
        Finset.sum_range_succ]
    · simp [hn,Nat.min_eq_right (show T≤n by omega),Nat.min_eq_right (show T≤n+1 by omega)]

theorem history_padded (c : Shape) (xs : Inputs ℝ) (head chunk i j : Nat) :
    (∑ u∈Finset.range (chunk*32), kv c xs head u i j)=history c xs head chunk i j :=
  sum_guard c.T (chunk*32) _

theorem output_padded (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head row j : Nat) :
    scale*(∑ u∈Finset.range ((row/32+1)*32),
      ∑ e∈Finset.range c.D, xs.q head row e*kv c xs head u e j)=output c xs scale head row j := by
  unfold output
  congr 1
  calc
    _ = ∑ u∈Finset.range ((row/32+1)*32),
        if u<c.T then ∑ e∈Finset.range c.D, xs.q head row e*xs.k head u e*xs.v head u j else 0 := by
      apply Finset.sum_congr rfl
      intro u hu
      by_cases h : u<c.T
      · simp [kv,h,mul_assoc]
      · simp [kv,h]
    _ = _ := sum_guard c.T _ _
end MathTarget

namespace RealProof
theorem carry_seed (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head n i j : Nat) (ifcond : Bool) :
    (if ifcond && decide (n=0) then realOps.zero else
      Reference.carry realOps c xs scale head ifcond n i j)=
      Reference.carry realOps c xs scale head ifcond n i j := by
  by_cases hn : n=0
  · subst n
    simp [Reference.carry,realOps]
  · simp [hn]

theorem kv_masked (c : Shape) (xs : Inputs ℝ) (head u i j : Nat) (hi : i<c.D) (hj : j<c.D) :
    masked realOps c xs.k head u i * masked realOps c xs.v head u j =
      MathTarget.kv c xs head u i j := by
  by_cases h : u<c.T <;> simp [masked,MathTarget.kv,h,hi,hj,realOps]

theorem carry_padded (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head : Nat) (ifcond : Bool)
    (n i j : Nat) (hi : i<c.D) (hj : j<c.D) :
    Reference.carry realOps c xs scale head ifcond n i j =
      ∑ u∈Finset.range (n*32), MathTarget.kv c xs head u i j := by
  induction n with
  | zero => simp [Reference.carry,realOps]
  | succ n ih =>
    simp only [Reference.carry,Body.evaluate,Body.nextCarry,chain_real,carry_seed]
    rw [ih,Nat.succ_mul,Finset.sum_range_add]
    congr 1
    apply Finset.sum_congr rfl
    intro u hu
    exact kv_masked c xs head (n*32+u) i j hi hj

theorem carry_target (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head : Nat) (ifcond : Bool)
    (n i j : Nat) (hi : i<c.D) (hj : j<c.D) :
    Reference.carry realOps c xs scale head ifcond n i j=MathTarget.history c xs head n i j := by
  rw [carry_padded c xs scale head ifcond n i j hi hj,MathTarget.history_padded]

theorem local_padded (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head chunk i j : Nat)
    (hr : chunk*32+i<c.T) (hj : j<c.D) :
    Body.localOutput realOps
      (Body.score realOps c (Body.query realOps c xs scale head chunk) (Body.key realOps c xs head chunk))
      (Body.value realOps c xs head chunk) i j =
      scale * ∑ t∈Finset.range 32, ∑ e∈Finset.range c.D,
        xs.q head (chunk*32+i) e * MathTarget.kv c xs head (chunk*32+t) e j := by
  rw [Body.localOutput,chain_real]
  simp only [show realOps.zero=(0:ℝ) from rfl,show realOps.cast=id from rfl,id_eq,zero_add]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro t ht
  rw [Body.score,chain_real]
  simp only [show realOps.zero=(0:ℝ) from rfl,zero_add]
  rw [Finset.sum_mul,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro e he
  have hd := Finset.mem_range.mp he
  by_cases hu : chunk*32+t<c.T
  · simp [Body.query,Body.key,Body.value,masked,MathTarget.kv,realOps,hr,hd,hj,hu]
    ring
  · simp [Body.query,Body.key,Body.value,masked,MathTarget.kv,realOps,hr,hd,hj,hu]

end RealProof

#specNonCircular MathTarget.output avoiding [Reference.output,Source.run]
#specNonCircular MathTarget.history avoiding [Reference.carry,Source.run]
#axiomsClean RealProof.carry_target
#axiomsClean RealProof.local_padded
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
