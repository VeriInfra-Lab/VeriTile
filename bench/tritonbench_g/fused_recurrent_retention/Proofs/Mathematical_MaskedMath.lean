import bench.tritonbench_g.fused_recurrent_retention.Proofs.Mathematical_Model
import VeriTile.Meta.StatementAudit

/-! Independent initial-memory formulas, including both feature tails.
The state variables in these formulas are finite sums, not execution results
or caller-supplied carry buffers. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.MaskedMath
open VeriTile.Triton
open scoped BigOperators
noncomputable section

def decayedPrefix (b seed : ℝ) (f : Nat → ℝ) (m : Nat) : ℝ :=
  seed * b ^ m + ∑ t ∈ Finset.range m, f t * b ^ (m - 1 - t)

theorem prefix_zero (b seed : ℝ) (f : Nat → ℝ) :
    decayedPrefix b seed f 0 = seed := by simp [decayedPrefix]

theorem prefix_succ (b seed : ℝ) (f : Nat → ℝ) (m : Nat) :
    decayedPrefix b seed f (m + 1) = b * decayedPrefix b seed f m + f m := by
  unfold decayedPrefix
  rw [Finset.sum_range_succ, show m + 1 - 1 - m = 0 by omega, pow_zero, mul_one]
  have hs : (∑ t ∈ Finset.range m, f t * b ^ (m + 1 - 1 - t)) =
      b * ∑ t ∈ Finset.range m, f t * b ^ (m - 1 - t) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro t ht
    have ht' := Finset.mem_range.mp ht
    rw [show m + 1 - 1 - t = (m - 1 - t) + 1 by omega, pow_succ]
    ring
  rw [hs, pow_succ]
  ring

def decayedSuffix (b : ℝ) (f : Nat → ℝ) (n t : Nat) : ℝ :=
  ∑ u ∈ Finset.Ico t n, f u * b ^ (u - t)

theorem suffix_end (b : ℝ) (f : Nat → ℝ) (n : Nat) :
    decayedSuffix b f n n = 0 := by simp [decayedSuffix]

theorem suffix_step (b : ℝ) (f : Nat → ℝ) (n t : Nat) (ht : t < n) :
    decayedSuffix b f n t = f t + b * decayedSuffix b f n (t + 1) := by
  unfold decayedSuffix
  rw [Finset.sum_eq_sum_Ico_succ_bot ht, Nat.sub_self, pow_zero, mul_one]
  congr 1
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro u hu
  have hu' := Finset.mem_Ico.mp hu
  rw [show u - t = (u - (t + 1)) + 1 by omega, pow_succ]
  ring

def decay (p : Layout) (l : Launch) : ℝ :=
  1 - Real.exp ((-5 - ((l.head % p.H : Nat) : ℝ)) * Real.log 2)

def keyValue (p : Layout) (l : Launch) (initial : BlockState) (r : RegionName)
    (t : Nat) (j : Fin p.BK) : ℝ :=
  if activeKey p l j.val then initial.readMem r (keyAddress p l t j.val) else 0

def valueValue (p : Layout) (l : Launch) (initial : BlockState) (r : RegionName)
    (t : Nat) (j : Fin p.BV) : ℝ :=
  if activeValue p l j.val then initial.readMem r (valueAddress p l t j.val) else 0

def seed (p : Layout) (l : Launch) (initial : BlockState) (r : RegionName)
    (hasSeed : Bool) (jk : Fin p.BK) (jv : Fin p.BV) : ℝ :=
  if hasSeed = Bool.true ∧ activeKey p l jk.val ∧ activeValue p l jv.val then
    initial.readMem r (stateAddress p l jk.val jv.val)
  else 0

def stateValue (p : Layout) (l : Launch) (initial : BlockState)
    (k v initialState : RegionName) (hasSeed : Bool) (m : Nat)
    (jk : Fin p.BK) (jv : Fin p.BV) : ℝ :=
  decayedPrefix (decay p l) (seed p l initial initialState hasSeed jk jv)
    (fun t => keyValue p l initial k t jk * valueValue p l initial v t jv) m

theorem state_zero (p : Layout) (l : Launch) (initial : BlockState)
    (k v initialState : RegionName) (hasSeed : Bool) (jk : Fin p.BK) (jv : Fin p.BV) :
    stateValue p l initial k v initialState hasSeed 0 jk jv =
      seed p l initial initialState hasSeed jk jv := by
  exact prefix_zero _ _ _

theorem state_succ (p : Layout) (l : Launch) (initial : BlockState)
    (k v initialState : RegionName) (hasSeed : Bool) (m : Nat)
    (jk : Fin p.BK) (jv : Fin p.BV) :
    stateValue p l initial k v initialState hasSeed (m + 1) jk jv =
      decay p l * stateValue p l initial k v initialState hasSeed m jk jv +
        keyValue p l initial k m jk * valueValue p l initial v m jv := by
  exact prefix_succ _ _ _ _

theorem state_inactive (p : Layout) (l : Launch) (initial : BlockState)
    (k v initialState : RegionName) (hasSeed : Bool) (m : Nat)
    (jk : Fin p.BK) (jv : Fin p.BV)
    (h : ¬ activeKey p l jk.val ∨ ¬ activeValue p l jv.val) :
    stateValue p l initial k v initialState hasSeed m jk jv = 0 := by
  rcases h with hk | hv
  · simp [stateValue, decayedPrefix, keyValue, seed, hk]
  · simp [stateValue, decayedPrefix, valueValue, seed, hv]

def output (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (jv : Fin p.BV) : ℝ :=
  ∑ jk : Fin p.BK,
    stateValue p l initial r.k r.v r.initial hasSeed (t + 1) jk jv *
      (keyValue p l initial r.q t jk * p.scale)

def dq (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (jk : Fin p.BK) : ℝ :=
  (∑ jv : Fin p.BV,
    stateValue p l initial r.k r.v r.initial hasSeed (t + 1) jk jv *
      valueValue p l initial r.do_ t jv) * p.scale

def dStateValue (p : Layout) (l : Launch) (initial : BlockState)
    (q do_ : RegionName) (t : Nat) (jk : Fin p.BK) (jv : Fin p.BV) : ℝ :=
  decayedSuffix (decay p l)
    (fun u => (keyValue p l initial q u jk * p.scale) * valueValue p l initial do_ u jv)
    p.T t

theorem dstate_end (p : Layout) (l : Launch) (initial : BlockState)
    (q do_ : RegionName) (jk : Fin p.BK) (jv : Fin p.BV) :
    dStateValue p l initial q do_ p.T jk jv = 0 := by exact suffix_end _ _ _

theorem dstate_step (p : Layout) (l : Launch) (initial : BlockState)
    (q do_ : RegionName) (t : Nat) (ht : t < p.T) (jk : Fin p.BK) (jv : Fin p.BV) :
    dStateValue p l initial q do_ t jk jv =
      (keyValue p l initial q t jk * p.scale) * valueValue p l initial do_ t jv +
        decay p l * dStateValue p l initial q do_ (t + 1) jk jv := by
  exact suffix_step _ _ _ _ ht

def dk (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (jk : Fin p.BK) : ℝ :=
  ∑ jv : Fin p.BV, dStateValue p l initial r.q r.do_ t jk jv *
    valueValue p l initial r.v t jv

def dv (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (jv : Fin p.BV) : ℝ :=
  ∑ jk : Fin p.BK, dStateValue p l initial r.q r.do_ t jk jv *
    keyValue p l initial r.k t jk

#axiomsClean prefix_zero
#axiomsClean prefix_succ
#axiomsClean suffix_end
#axiomsClean suffix_step
#axiomsClean state_zero
#axiomsClean state_succ
#axiomsClean state_inactive
#axiomsClean dstate_end
#axiomsClean dstate_step
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.MaskedMath
