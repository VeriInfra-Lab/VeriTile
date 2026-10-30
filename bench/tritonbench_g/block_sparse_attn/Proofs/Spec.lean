import bench.tritonbench_g.block_sparse_attn.Proofs.Mathematical_StreamingMath
import bench.tritonbench_g.flash_attn.Proofs.RealSum

namespace VeriTile.Bench.Repairs.BlockSparseAttn
open VeriTile.Triton
open scoped BigOperators
noncomputable section

structure Shape where
  batches : Nat
  heads : Nat
  kvHeads : Nat
  length : Nat
  blockM : Nat
  blockN : Nat
  blockD : Nat
  blocks : Nat
  layouts : Nat

def Shape.dim (s : Shape) := s.blockD * s.blocks
def Shape.Valid (s : Shape) : Prop :=
  0 < s.batches ∧ 0 < s.heads ∧ 0 < s.kvHeads ∧ s.kvHeads ∣ s.heads ∧
  0 < s.length ∧ 0 < s.blockM ∧ 0 < s.blockN ∧ 0 < s.blockD ∧
  0 < s.blocks ∧ 0 < s.layouts

structure Strides where
  batch : Nat
  head : Nat
  row : Nat
  dim : Nat
  deriving DecidableEq

def address (t : Strides) (b h n d : Nat) :=
  b*t.batch + h*t.head + n*t.row + d*t.dim
def signedAddress (t : Strides) (b h : Nat) (n : Int) (d : Nat) : Int :=
  (b*t.batch+h*t.head : Nat)+n*(t.row : Int)+(d*t.dim : Nat)
def outputAddress (s : Shape) (b h n d : Nat) := ((b*s.heads+h)*s.length+n)*s.dim+d
def kvHead (s : Shape) (h : Nat) := h/(s.heads/s.kvHeads)

structure CSR where
  rowStride : Nat
  colStride : Nat
  rows : Int → Int
  cols : Int → Int

def CSR.first (s : Shape) (c : CSR) (h row : Nat) : Int :=
  c.rows (((h%s.layouts)*c.rowStride+row/s.blockM : Nat) : Int)
def CSR.last (s : Shape) (c : CSR) (h row : Nat) : Int :=
  c.rows (((h%s.layouts)*c.rowStride+row/s.blockM+1 : Nat) : Int)
def CSR.count (s : Shape) (c : CSR) (h row : Nat) : Nat :=
  (c.last s h row-c.first s h row).toNat
def CSR.slot (s : Shape) (c : CSR) (h row visit : Nat) : Int :=
  ((h%s.layouts)*c.colStride : Nat) + c.first s h row + (visit : Int)
def CSR.key (s : Shape) (c : CSR) (h row visit lane : Nat) : Int :=
  c.cols (c.slot s h row visit)*(s.blockN : Int)+(lane : Int)

structure Inputs where
  query : Nat → Nat → Nat → Nat → ℝ
  key : Nat → Nat → Int → Nat → ℝ
  value : Nat → Nat → Int → Nat → ℝ

namespace Spec
def dot (s : Shape) (x : Inputs) (b h row : Nat) (key : Int) : ℝ :=
  ((List.range s.dim).map (fun d=>x.query b h row d*x.key b (kvHead s h) key d)).sum
def score (s : Shape) (c : CSR) (x : Inputs) (scale : ℝ)
    (b h row visit : Nat) (lane : Fin s.blockN) : WithBot ℝ :=
  let key := c.key s h row visit lane.val
  if key ≤ (row : Int) then some (dot s x b h row key*scale) else none
def weight (s : Shape) (c : CSR) (x : Inputs) (scale : ℝ)
    (b h row visit : Nat) (lane : Fin s.blockN) : ℝ :=
  let key := c.key s h row visit lane.val
  if key ≤ (row : Int) then Real.exp (dot s x b h row key*scale) else 0
def output (s : Shape) (c : CSR) (x : Inputs) (scale : ℝ) (b h row col : Nat) : ℝ :=
  (∑ visit ∈ Finset.range (c.count s h row), ∑ lane : Fin s.blockN,
    weight s c x scale b h row visit lane * x.value b (kvHead s h) (c.key s h row visit lane.val) col) /
  (∑ visit ∈ Finset.range (c.count s h row), ∑ lane : Fin s.blockN,
    weight s c x scale b h row visit lane)

/-- The repaired score loop visits all channel blocks, including blocks after two. -/
def scoreLoop (s : Shape) (x : Inputs) (b h row : Nat) (key : Int) : ℝ :=
  (List.range s.blocks).foldl (fun acc block => acc +
    FlashAttn.RealSum.sumReal (fun d=>
      x.query b h row (block*s.blockD+d)*x.key b (kvHead s h) key (block*s.blockD+d)) s.blockD) 0

theorem scoreLoop_full (s : Shape) (x : Inputs) (b h row : Nat) (key : Int) :
    scoreLoop s x b h row key=dot s x b h row key := by
  rw [scoreLoop,FlashAttn.RealSum.range_fold,zero_add]
  rw [FlashAttn.RealSum.sum_chunks
    (fun d=>x.query b h row d*x.key b (kvHead s h) key d) s.blockD s.blocks,
    FlashAttn.RealSum.sumReal_list]
  simp only [dot,Shape.dim,Nat.mul_comm]

/-- Reuses the already proved source-shaped normalized update, with the new
full-D score and unchanged signed CSR occurrence sequence. -/
def recurrence (s : Shape) (c : CSR) (x : Inputs) (scale : ℝ) (b h row col : Nat) : ℝ :=
  TritonBenchG.BlockSparseAttn.StreamingMath.accumulated (score s c x scale b h row)
    (fun visit lane=>x.value b (kvHead s h) (c.key s h row visit lane.val) col) (c.count s h row)

theorem recurrence_output (s : Shape) (c : CSR) (x : Inputs) (scale : ℝ) (b h row col : Nat) :
    recurrence s c x scale b h row col=output s c x scale b h row col := by
  rw [recurrence,TritonBenchG.BlockSparseAttn.StreamingMath.accumulated_eq_formula]
  have hw (visit : Nat) (lane : Fin s.blockN) :
      TritonBenchG.BlockSparseAttn.StreamingMath.weight (score s c x scale b h row visit lane)=
        weight s c x scale b h row visit lane := by
    unfold score weight
    dsimp only
    split <;> rfl
  simp only [TritonBenchG.BlockSparseAttn.StreamingMath.moment,
    TritonBenchG.BlockSparseAttn.StreamingMath.mass,output,hw]

theorem empty_output (s : Shape) (c : CSR) (x : Inputs) (scale : ℝ) (b h row col : Nat)
    (empty : c.last s h row ≤ c.first s h row) : output s c x scale b h row col=0 := by
  have hz : c.count s h row=0 := by unfold CSR.count;omega
  simp [output,hz]
#axiomsClean scoreLoop_full
#axiomsClean recurrence_output
#axiomsClean empty_output
end Spec
end
end VeriTile.Bench.Repairs.BlockSparseAttn
