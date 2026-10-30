import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Source
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Ring
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open scoped BigOperators

namespace MathTarget

def headRate (c : Shape) (head : Nat) : ℝ :=
  1-Real.exp ((-5-(head%c.H : Nat) : ℝ)*Real.log 2)

def weighted (rate seed : ℝ) (term : Nat → ℝ) (time : Nat) : ℝ :=
  rate^time*seed+∑ u ∈ Finset.range time, rate^(time-1-u)*term u

/-- Independent suffix sum: adjoint at a time boundary. -/
def suffix (rate terminal : ℝ) (term : Nat → ℝ) (start count : Nat) : ℝ :=
  rate^count*terminal + ∑ i ∈ Finset.range count, rate^(i+1)*term (start+i)

/-- Coefficient of the token's outer product in the complete loss. -/
def coefficient (rate terminal : ℝ) (term : Nat → ℝ) (start count : Nat) : ℝ :=
  rate^(count-1)*terminal + ∑ i ∈ Finset.range count, rate^i*term (start+i)

def state (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head time key value : Nat) : ℝ :=
  weighted (headRate c head) (if hasSeed then xs.seed head key value else 0)
    (fun u => xs.k head u key*xs.v head u value) time

def forcing (xs : Inputs ℝ) (scale : ℝ) (head key value time : Nat) :=
  scale*xs.q head time key*xs.incoming head time value

def future (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head time key value : Nat) :=
  suffix (headRate c head) (xs.finalIncoming head key value)
    (forcing xs scale head key value) time (c.T-time)

def tokenCoefficient (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head time key value : Nat) :=
  coefficient (headRate c head) (xs.finalIncoming head key value)
    (forcing xs scale head key value) time (c.T-time)

def output (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (hasSeed : Bool)
    (head row value : Nat) : ℝ :=
  scale*∑ key ∈ Finset.range c.K,
    xs.q head row key*state c xs hasSeed head (row+1) key value

def finalState (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head key value : Nat) : ℝ := state c xs hasSeed head c.T key value

def queryGradient (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (hasSeed : Bool)
    (head time key : Nat) : ℝ :=
  scale*∑ value ∈ Finset.range c.V,
    xs.incoming head time value*state c xs hasSeed head (time+1) key value

def keyGradient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head time key : Nat) : ℝ :=
  ∑ value ∈ Finset.range c.V,
    xs.v head time value*tokenCoefficient c xs scale head time key value

def valueGradient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head time value : Nat) : ℝ :=
  ∑ key ∈ Finset.range c.K,
    xs.k head time key*tokenCoefficient c xs scale head time key value

def initialGradient (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head key value : Nat) : ℝ := future c xs scale head 0 key value

def loss (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (hasSeed : Bool) : ℝ :=
  (∑ head ∈ Finset.range c.heads, ∑ row ∈ Finset.range c.T, ∑ value ∈ Finset.range c.V,
    xs.incoming head row value*output c xs scale hasSeed head row value)+
  ∑ head ∈ Finset.range c.heads, ∑ key ∈ Finset.range c.K, ∑ value ∈ Finset.range c.V,
    xs.finalIncoming head key value*finalState c xs hasSeed head key value

@[simp] theorem weighted_zero (rate seed : ℝ) (term : Nat → ℝ) : weighted rate seed term 0=seed := by
  simp [weighted]

theorem weighted_succ (rate seed : ℝ) (term : Nat → ℝ) (time : Nat) :
    weighted rate seed term (time+1)=rate*weighted rate seed term time+term time := by
  unfold weighted
  rw [Finset.sum_range_succ]
  have hsum : (∑ u ∈ Finset.range time, rate^(time+1-1-u)*term u)=
      rate*∑ u ∈ Finset.range time, rate^(time-1-u)*term u := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro u hu
    have h : time+1-1-u=(time-1-u)+1 := by have := Finset.mem_range.mp hu; omega
    rw [h,pow_succ]
    ring
  rw [hsum,pow_succ]
  simp only [Nat.add_sub_cancel,Nat.sub_self,pow_zero,one_mul]
  ring

theorem weighted_add (rate seed : ℝ) (term : Nat → ℝ) (start count : Nat) :
    weighted rate seed term (start+count)=
      weighted rate (weighted rate seed term start) (fun i => term (start+i)) count := by
  induction count with
  | zero => simp
  | succ n ih => rw [Nat.add_succ,weighted_succ,ih,weighted_succ]

@[simp] theorem suffix_zero (rate terminal : ℝ) (term : Nat → ℝ) (start : Nat) :
    suffix rate terminal term start 0=terminal := by simp [suffix]

theorem suffix_split (rate terminal : ℝ) (term : Nat → ℝ) (start first rest : Nat) :
    suffix rate terminal term start (first+rest)=
      rate^first*suffix rate terminal term (start+first) rest+
      ∑ i ∈ Finset.range first, rate^(i+1)*term (start+i) := by
  unfold suffix
  rw [Finset.sum_range_add,pow_add]
  have h : (∑ i ∈ Finset.range rest, rate^(first+i+1)*term (start+(first+i)))=
      rate^first*∑ i ∈ Finset.range rest, rate^(i+1)*term (start+first+i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    rw [show first+i+1=first+(i+1) by omega,pow_add]
    simp only [Nat.add_assoc]
    ring
  rw [h]
  ring

theorem coefficient_split (rate terminal : ℝ) (term : Nat → ℝ) (start first rest : Nat)
    (hf : 0<first) :
    coefficient rate terminal term start (first+rest)=
      rate^(first-1)*suffix rate terminal term (start+first) rest+
      ∑ i ∈ Finset.range first, rate^i*term (start+i) := by
  unfold coefficient suffix
  rw [Finset.sum_range_add]
  have hp : first+rest-1=(first-1)+rest := by omega
  rw [hp,pow_add]
  have h : (∑ i ∈ Finset.range rest, rate^(first+i)*term (start+(first+i)))=
      rate^(first-1)*∑ i ∈ Finset.range rest, rate^(i+1)*term (start+first+i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    have he : rate^(first+i)=rate^(first-1)*rate^(i+1) := by
      rw [← pow_add]
      congr 1
      omega
    rw [he]
    simp only [Nat.add_assoc]
    ring
  rw [h]
  ring

theorem state_chunk (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool)
    (head chunk key value : Nat) (hc : chunk<c.chunks) :
    state c xs hasSeed head (c.boundary (chunk+1)) key value=
      (headRate c head)^(c.length chunk)*state c xs hasSeed head (c.boundary chunk) key value+
      ∑ i ∈ Finset.range (c.length chunk), (headRate c head)^(c.length chunk-1-i)*
        (xs.k head (chunk*64+i) key*xs.v head (chunk*64+i) value) := by
  rw [c.boundary_succ chunk hc,c.boundary_at_start chunk hc]
  exact weighted_add (headRate c head) (if hasSeed then xs.seed head key value else 0)
    (fun u => xs.k head u key*xs.v head u value) (chunk*64) (c.length chunk)

theorem future_chunk (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk key value : Nat) (hc : chunk<c.chunks) :
    future c xs scale head (c.boundary chunk) key value=
      (headRate c head)^(c.length chunk)*future c xs scale head (c.boundary (chunk+1)) key value+
      ∑ i ∈ Finset.range (c.length chunk), (headRate c head)^(i+1)*
        forcing xs scale head key value (chunk*64+i) := by
  rw [c.boundary_at_start chunk hc,c.boundary_succ chunk hc]
  unfold future
  have hl := c.length_le chunk
  have hs := c.chunk_start chunk hc
  have hb : chunk*64+c.length chunk≤c.T := by dsimp [Shape.length]; omega
  have h : c.T-chunk*64=c.length chunk+(c.T-(chunk*64+c.length chunk)) := by omega
  rw [h]
  exact suffix_split _ _ _ _ _ _

@[simp] theorem future_final (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head key value : Nat) :
    future c xs scale head c.T key value=xs.finalIncoming head key value := by simp [future]

theorem token_chunk (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk lane key value : Nat) (hc : chunk<c.chunks) (hl : lane<c.length chunk) :
    tokenCoefficient c xs scale head (chunk*64+lane) key value=
      (headRate c head)^(c.length chunk-1-lane)*
        future c xs scale head (c.boundary (chunk+1)) key value+
      ∑ i ∈ Finset.range (c.length chunk-lane), (headRate c head)^i*
        forcing xs scale head key value (chunk*64+lane+i) := by
  rw [c.boundary_succ chunk hc]
  unfold tokenCoefficient future
  have hs := c.chunk_start chunk hc
  have hb : chunk*64+c.length chunk≤c.T := by dsimp [Shape.length]; omega
  have h : c.T-(chunk*64+lane)=(c.length chunk-lane)+(c.T-(chunk*64+c.length chunk)) := by omega
  rw [h,coefficient_split _ _ _ _ _ _ (by omega)]
  have h1 : c.length chunk-lane-1=c.length chunk-1-lane := by omega
  have h2 : chunk*64+lane+(c.length chunk-lane)=chunk*64+c.length chunk := by omega
  rw [h1,h2]

/-- Local chain rule; summing this identity telescopes the complete reverse pass.
All quantities are freely quantified real values, not numerical oracle outputs. -/
theorem local_vjp (rate q k v state dq dk dv ds incoming terminal scale : ℝ) :
    incoming*(scale*(dq*(rate*state+k*v)+q*(rate*ds+dk*v+k*dv)))+
      terminal*(rate*ds+dk*v+k*dv)=
    (scale*incoming*(rate*state+k*v))*dq+
      ((terminal+scale*q*incoming)*v)*dk+
      ((terminal+scale*q*incoming)*k)*dv+
      (rate*terminal+rate*scale*q*incoming)*ds := by ring

#specNonCircular output avoiding [weighted_succ,state_chunk]
#specNonCircular queryGradient avoiding [weighted_succ,state_chunk]
#specNonCircular keyGradient avoiding [future_chunk,token_chunk]
#specNonCircular valueGradient avoiding [future_chunk,token_chunk]
#specNonCircular initialGradient avoiding [future_chunk,token_chunk]
#axiomsClean state_chunk
#axiomsClean future_chunk
#axiomsClean token_chunk
#axiomsClean local_vjp
end MathTarget
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
