import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.VJP
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.AddressSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open scoped BigOperators

namespace ScalarAlgebra

theorem tiled_sum (f : Nat → ℝ) (blocks tile : Nat) :
    (∑ b ∈ Finset.range blocks, ∑ lane ∈ Finset.range tile, f (b*tile+lane))=
      ∑ i ∈ Finset.range (blocks*tile), f i := by
  induction blocks with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]

theorem padded_sum (f : Nat → ℝ) (size tile : Nat) (ht : 0<tile) :
    (∑ b ∈ Finset.range ((size+tile-1)/tile), ∑ lane ∈ Finset.range tile,
      if b*tile+lane<size then f (b*tile+lane) else 0)=∑ i ∈ Finset.range size, f i := by
  rw [tiled_sum (fun i => if i<size then f i else 0)]
  exact (MathTarget.sum_prefix f size _ (Shape.block_cover size tile ht)).symm

/-- Scalarization of the two nested dot products in fwd_o, dq and dk.
Only ring laws and finite reindexing are used in the precision-erased profile. -/
theorem contraction (outer inner : Finset Nat) (a b : Nat → ℝ) (c : Nat → Nat → ℝ)
    (weight : Nat → ℝ) (decay : ℝ) :
    (∑ i ∈ outer, a i*b i)*decay+
      (∑ j ∈ inner, (∑ i ∈ outer, a i*c j i)*weight j)=
    ∑ i ∈ outer, a i*(decay*b i+∑ j ∈ inner, weight j*c j i) := by
  simp_rw [mul_add,Finset.sum_add_distrib,Finset.mul_sum,Finset.sum_mul]
  rw [Finset.sum_comm]
  congr 1
  · apply Finset.sum_congr rfl
    intro i hi
    ring
  · apply Finset.sum_congr rfl
    intro i hi
    apply Finset.sum_congr rfl
    intro j hj
    ring

/-- All sources preserve the same finite sum after the head/key/value tile
coordinates have been zero-extended and flattened. -/
theorem contraction_scaled (outer inner : Finset Nat) (a b : Nat → ℝ)
    (c : Nat → Nat → ℝ) (weight : Nat → ℝ) (decay scale : ℝ) :
    ((∑ i ∈ outer, (a i*decay)*b i)+
      (∑ j ∈ inner, (∑ i ∈ outer, a i*c j i)*weight j))*scale=
    scale*∑ i ∈ outer, a i*(decay*b i+∑ j ∈ inner, weight j*c j i) := by
  have h : (∑ i ∈ outer, (a i*decay)*b i)=(∑ i ∈ outer, a i*b i)*decay := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [h,contraction]
  ring
end ScalarAlgebra

namespace ScalarSource
open MathTarget

/-- Exact real interpretation of the forward-state source recurrence.
The native layer below separately keeps every actual rounded scalar node. -/
def forwardCarry (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool) (head : Nat) :
    Nat → Nat → Nat → ℝ
  | 0, key, value => if hasSeed then xs.seed head key value else 0
  | chunk+1, key, value =>
      (headRate c head)^(c.length chunk)*forwardCarry c xs hasSeed head chunk key value+
      ∑ lane ∈ Finset.range (c.length chunk), xs.k head (chunk*64+lane) key*
        (xs.v head (chunk*64+lane) value*(headRate c head)^(c.length chunk-1-lane))

/-- The reverse source stores this carry before updating the current chunk. -/
def reverseCarry (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head : Nat) :
    Nat → Nat → Nat → ℝ
  | 0, key, value => xs.finalIncoming head key value
  | step+1, key, value =>
      let chunk := c.chunks-1-step
      (headRate c head)^(c.length chunk)*reverseCarry c xs scale head step key value+
      ∑ lane ∈ Finset.range (c.length chunk), (xs.q head (chunk*64+lane) key*scale)*
        (xs.incoming head (chunk*64+lane) value*(headRate c head)^(lane+1))

theorem forward_carry_correct (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool) (head chunk : Nat)
    (hc : chunk≤c.chunks) (key value : Nat) :
    forwardCarry c xs hasSeed head chunk key value=
      state c xs hasSeed head (c.boundary chunk) key value := by
  induction chunk with
  | zero => simp [forwardCarry,state,Shape.boundary]
  | succ n ih =>
    have hn : n<c.chunks := by omega
    rw [forwardCarry,ih (by omega),state_chunk c xs hasSeed head n key value hn]
    congr 1
    apply Finset.sum_congr rfl
    intro lane hl
    ring

theorem reverse_carry_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head step : Nat)
    (hs : step≤c.chunks) (key value : Nat) :
    reverseCarry c xs scale head step key value=
      future c xs scale head (c.boundary (c.chunks-step)) key value := by
  induction step with
  | zero => simp [reverseCarry,c.boundary_final]
  | succ n ih =>
    have hn : n<c.chunks := by omega
    have hc : c.chunks-1-n<c.chunks := by omega
    have hnext : c.chunks-(n+1)=c.chunks-1-n := by omega
    have hprev : c.chunks-n=(c.chunks-1-n)+1 := by omega
    rw [reverseCarry,ih (by omega),hnext,hprev,
      future_chunk c xs scale head (c.chunks-1-n) key value hc]
    congr 1
    apply Finset.sum_congr rfl
    intro lane hl
    unfold forcing
    ring

theorem history_correct (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool) (head chunk key value : Nat)
    (hc : chunk<c.chunks) :
    forwardCarry c xs hasSeed head chunk key value=state c xs hasSeed head (chunk*64) key value := by
  rw [forward_carry_correct c xs hasSeed head chunk (Nat.le_of_lt hc),c.boundary_at_start chunk hc]

theorem final_state_correct (c : Shape) (xs : Inputs ℝ) (hasSeed : Bool) (head key value : Nat) :
    forwardCarry c xs hasSeed head c.chunks key value=finalState c xs hasSeed head key value := by
  rw [forward_carry_correct c xs hasSeed head c.chunks le_rfl,c.boundary_final]
  rfl

theorem history_gradient_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (head chunk key value : Nat) (hc : chunk<c.chunks) :
    reverseCarry c xs scale head (c.chunks-1-chunk) key value=
      future c xs scale head (c.boundary (chunk+1)) key value := by
  rw [reverse_carry_correct c xs scale head (c.chunks-1-chunk) (by omega)]
  congr 2
  congr 1
  omega

theorem initial_gradient_correct (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (head key value : Nat) :
    reverseCarry c xs scale head c.chunks key value=initialGradient c xs scale head key value := by
  rw [reverse_carry_correct c xs scale head c.chunks le_rfl]
  simp [initialGradient,Shape.boundary]

#axiomsClean forward_carry_correct
#axiomsClean reverse_carry_correct
#axiomsClean initial_gradient_correct
end ScalarSource
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
