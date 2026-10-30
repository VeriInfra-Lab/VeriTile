import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_WholeGrid
import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_Contract

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section

def realSum {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),
      xs (TileShape.insertAxisIndex shape ax out j)
  · exact fun out => ∑ j : Fin (TileShape.axisDim shape ax),
      xs (TileShape.replaceAxisIndex shape ax out j)

/-- Exact Real interpretation; all source casts and reduction operations are
retained in the source program. This is not a second native implementation. -/
def realAlgebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .sqrt => Real.sqrt x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} ax keep xs => realSum ax keep xs

theorem sum_prefix (N B : Nat) (complete : N ≤ B) (f : Nat → ℝ) :
    (∑ j : Fin B, if j.val < N then f j.val else 0) = ∑ j ∈ Finset.range N, f j := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j < N then f j else 0) B]
  calc
    (∑ j ∈ Finset.range B, if j < N then f j else 0) =
        ∑ j ∈ Finset.range N, if j < N then f j else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono complete)
      intro j _ hj
      simp only [Finset.mem_range] at hj
      simp [hj]
    _ = _ := Finset.sum_congr rfl (fun j hj => if_pos (Finset.mem_range.mp hj))

theorem sum_prefix_any (N B : Nat) (I : Fintype (Fin B)) (complete : N ≤ B) (f : Nat → ℝ) :
    @Finset.sum (Fin B) ℝ _ (@Finset.univ _ I) (fun j => if j.val < N then f j.val else 0) =
      ∑ j ∈ Finset.range N, f j := by
  have same : I = Fin.fintype B := Subsingleton.elim _ _
  rw [same]
  exact sum_prefix N B complete f

theorem rowValue_real (c : Layout) (eps : ℝ) (xs ws : Nat → ℝ) (i : Nat) :
    rowValue realAlgebra c eps xs ws i = realFormula c.N eps xs ws i := by
  have square : (fun j : Fin c.B =>
      (if j.val < c.N then xs j.val else 0) * (if j.val < c.N then xs j.val else 0)) =
      (fun j : Fin c.B => if j.val < c.N then xs j.val * xs j.val else 0) := by
    funext j
    split_ifs <;> simp
  simp only [rowValue, inputTile, realAlgebra, realSum, id_eq,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, realFormula]
  norm_num only
  rw [square]
  apply congrArg (fun total : ℝ => xs i * (1 / Real.sqrt (total / (c.N : ℝ) + eps)) * ws i)
  exact sum_prefix_any c.N c.B _ c.complete (fun j => xs j * xs j)

theorem regular_of_positive_epsilon (N : Nat) (eps : ℝ) (xs : Nat → ℝ)
    (positive : 0 < N) (heps : 0 < eps) : RegularRow N eps xs := by
  refine ⟨positive,?_⟩
  have sumNonnegative : 0 ≤ ∑ j ∈ Finset.range N, xs j * xs j :=
    Finset.sum_nonneg (fun j _ => mul_self_nonneg (xs j))
  have meanNonnegative := div_nonneg sumNonnegative (Nat.cast_nonneg N : (0 : ℝ) ≤ N)
  linarith

#guard_msgs (drop info) in
#specNonCircular realFormula avoiding [Layout.source,sourceRun,rowExec,runRows]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
