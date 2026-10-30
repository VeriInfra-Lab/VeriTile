import bench.tritonbench_g.l2_norm_bwd.Contract
import bench.tritonbench_g.l2_norm_bwd.Proofs.Scalar
import bench.tritonbench_g.l2_norm_bwd.Proofs.Mathematical_Math
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.L2NormBwd.Math
open VeriTile.Triton FP.Structural
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 8000000

def sumValues {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (values : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),values (TileShape.insertAxisIndex shape ax out j)
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),values (TileShape.replaceAxisIndex shape ax out j)

def algebra : Algebra ℝ where
  literal := fun _ _ r => r
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b | .max => max a b | .pow => a^b
  unary := fun _ op a => match op with | .sqrt => Real.sqrt a | _ => 0
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ => sumValues

theorem padded_sum (N B : Nat) (h : N≤B) (f : Nat → ℝ) :
    (∑ j : Fin B,if j.val<N then f j.val else 0)=∑ j : Fin N,f j.val := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j<N then f j else 0),Fin.sum_univ_eq_sum_range f]
  rw [show B=N+(B-N) by omega,Finset.sum_range_add]
  have first : (∑ j ∈ Finset.range N,if j<N then f j else 0)=∑ j ∈ Finset.range N,f j := by
    apply Finset.sum_congr rfl
    intro j hj
    exact if_pos (Finset.mem_range.mp hj)
  rw [first]
  have tail : (∑ j ∈ Finset.range (B-N),if N+j<N then f (N+j) else 0)=0 := by
    apply Finset.sum_eq_zero
    intro j _
    exact if_neg (by omega)
  rw [tail,add_zero]

theorem productSum_real (N B : Nat) (h : N≤B) (xs ys : Nat → ℝ) :
    Tile.productSum algebra N B xs ys=∑ j : Fin N,xs j.val*ys j.val := by
  have padded : Tile.productSum algebra N B xs ys=
      ∑ j : Fin B,if j.val<N then xs j.val*ys j.val else 0 := by
    simp only [Tile.productSum,algebra,sumValues,TileShape.axisDim,TileShape.insertAxisIndex,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    apply Finset.sum_congr rfl
    intro j _
    by_cases active : j.val<N <;> simp [Tile.masked,Tile.loaded,active,algebra]
  rw [padded]
  exact padded_sum N B h (fun j => xs j*ys j)

theorem source_derivative (N B : Nat) (h : N≤B) (eps : ℝ) (xs dys : Nat → ℝ) (i : Nat) :
    Tile.scalar algebra N B eps xs dys i=derivative N eps xs dys i := by
  unfold Tile.scalar
  rw [productSum_real N B h xs xs,productSum_real N B h dys xs]
  rfl

theorem reused_original_formula (N B : Nat) (h : N≤B) (eps : ℝ)
    (xs dys : Nat → ℝ) (i : Fin B) (active : i.val<N) :
    VeriTile.Bench.TritonBenchG.L2NormBwd.rowFormula N B eps
      (fun j => xs j.val) (fun j => dys j.val) i=derivative N eps xs dys i.val := by
  unfold VeriTile.Bench.TritonBenchG.L2NormBwd.rowFormula derivative
  simp only [padded_sum N B h (fun j => xs j*xs j),padded_sum N B h (fun j => dys j*xs j)]

#specNonCircular derivative avoiding [Tile.scalar,algebra,source]
#axiomsClean source_derivative
#axiomsClean reused_original_formula
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Math
