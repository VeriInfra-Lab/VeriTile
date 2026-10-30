import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeSource
import Mathlib.Algebra.BigOperators.Intervals

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeMath
open VeriTile.Triton FP.Structural WholeSource
open Typed (Parameters)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Kernel-independent matrix-vector multiplication on logical coordinates. -/
def matrixVector (N M : Nat) (matrix : Fin N → Fin M → ℝ) (vector : Fin M → ℝ)
    (i : Fin N) : ℝ := ∑ j, matrix i j*vector j

def sumValues {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (values : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun i => ∑ j, values (TileShape.insertAxisIndex shape ax i j)
  · exact fun i => ∑ j, values (TileShape.replaceAxisIndex shape ax i j)

/-- Ideal Real interpretation of the complete original FP32 source.
It is a mathematical abstraction and makes no native floating equality claim. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y
    | .mul => x*y
    | _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} => sumValues

def product (p : Parameters) (memory : RegionName → Nat → Cell ℝ) (r k : Nat) : ℝ :=
  (memory p.a (r*p.strideRow+k*p.strideColumn)).read .fp32 *
    (memory p.b (k*p.strideVector)).read .fp32

def term (p : Parameters) (memory : RegionName → Nat → Cell ℝ) (r k : Nat) : ℝ :=
  if k<p.columns then product p memory r k else 0

def tail (p : Parameters) (memory : RegionName → Nat → Cell ℝ) (r off : Nat) : ℝ :=
  ∑ k ∈ Finset.range (p.columns-off), term p memory r (off+k)

theorem split_zero_extension (f : Nat → ℝ) (m b : Nat) (zero : ∀ k, m≤k → f k=0) :
    (∑ k ∈ Finset.range m, f k)=
      (∑ k ∈ Finset.range b, f k)+(∑ k ∈ Finset.range (m-b),f (b+k)) := by
  have hle : m≤b+(m-b) := by omega
  have he : (∑ k ∈ Finset.range m,f k)=(∑ k ∈ Finset.range (b+(m-b)),f k) := by
    apply Finset.sum_subset (Finset.range_mono hle)
    intro k hk notMem
    apply zero
    simpa using notMem
  rw [he,Finset.sum_range_add]

theorem tail_split (p : Parameters) (memory : RegionName → Nat → Cell ℝ) (r off : Nat) :
    tail p memory r off=(∑ j : Fin p.blockColumns,term p memory r (off+j.val))+
      tail p memory r (off+p.blockColumns) := by
  unfold tail
  rw [split_zero_extension (fun k => term p memory r (off+k)) (p.columns-off) p.blockColumns
    (by intro k hk; simp [term,show ¬ off+k<p.columns by omega])]
  have hn : p.columns-off-p.blockColumns=p.columns-(off+p.blockColumns) := by omega
  rw [hn]
  congr 1
  · exact (Fin.sum_univ_eq_sum_range _ _).symm
  · apply Finset.sum_congr rfl
    intro k hk
    congr 1
    omega

theorem next_value (p : Parameters) (pid off : Nat) (memory : RegionName → Nat → Cell ℝ)
    (acc : Acc ℝ p) (i : Fin p.blockRows) (active : row p pid i.val<p.rows) (j : Fin p.blockColumns) :
    nextAcc algebra p pid off memory acc (i,j,PUnit.unit)=
      acc (i,j,PUnit.unit)+term p memory (row p pid i.val) (off+j.val) := by
  simp only [nextAcc,aChunk,bChunk,load,algebra,active,true_and]
  by_cases h : off+j.val<p.columns
  · simp only [h,if_true,term,product]
    congr 2
    · congr 1
      unfold aOffset
      ring
    · congr 1
      unfold bOffset
      ring
  · simp [h,term]

theorem accumulated_sum (p : Parameters) (width : 0<p.blockColumns) (pid off : Nat)
    (memory : RegionName → Nat → Cell ℝ) (acc : Acc ℝ p)
    (i : Fin p.blockRows) (active : row p pid i.val<p.rows) :
    (∑ j : Fin p.blockColumns,accumulated algebra p pid memory off acc (i,j,PUnit.unit))=
      (∑ j : Fin p.blockColumns,acc (i,j,PUnit.unit))+tail p memory (row p pid i.val) off := by
  conv_lhs => rw [accumulated]
  by_cases h : off<p.columns
  · simp only [dif_pos (And.intro h width)]
    rw [accumulated_sum p width pid (off+p.blockColumns) memory _ i active]
    simp_rw [next_value p pid off memory acc i active]
    rw [Finset.sum_add_distrib,tail_split p memory (row p pid i.val) off]
    ring
  · simp [h,tail,show p.columns-off=0 by omega]
termination_by p.columns-off

theorem tile_mathematical (p : Parameters) (width : 0<p.blockColumns) (pid : Nat)
    (memory : RegionName → Nat → Cell ℝ) (i : Fin p.blockRows)
    (active : row p pid i.val<p.rows) :
    tileValue algebra p pid memory i=
      ∑ k : Fin p.columns, product p memory (row p pid i.val) k.val := by
  unfold tileValue storedValue outputValues
  change (∑ j : Fin p.blockColumns,
    accumulated algebra p pid memory 0 (fun _ => 0) (i,j,PUnit.unit))=_
  rw [accumulated_sum p width pid 0 memory _ i active]
  simp only [Finset.sum_const_zero,zero_add,tail,Nat.sub_zero,Nat.zero_add]
  rw [Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro k hk
  exact if_pos (Finset.mem_range.mp hk)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeMath
