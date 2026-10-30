import bench.tritonbench_g.l2_norm_triton1.Proofs.Scalar
import bench.tritonbench_g.l2_norm_triton1.Proofs.Mathematical_Math

namespace VeriTile.Bench.Repairs.L2NormTriton1.RealExecution
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

/-- Independent logical L2 normalization, over exactly the complete last
dimension. Real sqrt and division are total; arbitrary real epsilon is allowed. -/
def formula (N : Nat) (eps : ℝ) (xs : Nat → ℝ) (i : Nat) : ℝ :=
  xs i / Real.sqrt ((∑ j∈Finset.range N,(xs j)^2)+eps)

def sumValues {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (values : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ k : Fin (TileShape.axisDim shape ax),
      values (TileShape.insertAxisIndex shape ax out k)
  · exact fun out => ∑ k : Fin (TileShape.axisDim shape ax),
      values (TileShape.replaceAxisIndex shape ax out k)

def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .mul => x*y | .add => x+y | .sub => x-y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .sqrt => Real.sqrt x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ => sumValues

theorem variance_real (N B : Nat) (xs : Nat → ℝ) (cover : N≤B) :
    Tile.variance algebra N B xs=∑ j∈Finset.range N,(xs j)^2 := by
  have maskedSquare : (fun i : Fin B =>
      (if i.val<N then xs i.val else 0)*(if i.val<N then xs i.val else 0))=
      (fun i : Fin B => if i.val<N then (xs i.val)^2 else 0) := by
    funext i
    by_cases h : i.val<N <;> simp [h,pow_two]
  have filter : (Finset.range B).filter (fun i => i<N)=Finset.range N := by
    ext i
    simp only [Finset.mem_filter,Finset.mem_range]
    omega
  simp only [Tile.variance,Tile.masked,Tile.loaded,algebra,sumValues,
    TileShape.axisDim,TileShape.insertAxisIndex,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  trans ∑ j : Fin B,if j.val<N then (xs j.val)^2 else 0
  · apply Finset.sum_congr rfl
    intro j hj
    by_cases h : j.val<N <;> norm_num [h,pow_two]
  · change (∑ j : Fin B,(fun k : Nat => if k<N then (xs k)^2 else 0) j.val)=_
    rw [Fin.sum_univ_eq_sum_range (fun k : Nat => if k<N then (xs k)^2 else 0)]
    rw [←Finset.sum_filter,filter]

theorem scalar_real (N B : Nat) (eps : ℝ) (xs : Nat → ℝ) (i : Nat) (cover : N≤B) :
    Tile.scalar algebra N B eps xs i=formula N eps xs i := by
  unfold Tile.scalar
  rw [variance_real N B xs cover]
  simp only [algebra,formula]
  ring

/-- Existing independent padded L2 mathematics is reused without importing
an old public entry or asserting correctness of the original host wrapper. -/
theorem agrees_with_original_math (N B : Nat) (eps : ℝ) (xs : Fin B → ℝ)
    (i : Fin B) (hi : i.val<N) :
    VeriTile.Triton.TiledL2Norm.l2Norm (fun j => if j.val<N then xs j else 0) eps i=
      VeriTile.Bench.TritonBenchG.L2NormTriton1.rowFormula N B eps xs i :=
  VeriTile.Bench.TritonBenchG.L2NormTriton1.padded_formula N B eps xs i hi

#specNonCircular formula avoiding [Tile.scalar,Tile.variance,algebra]
#axiomsClean scalar_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.RealExecution
