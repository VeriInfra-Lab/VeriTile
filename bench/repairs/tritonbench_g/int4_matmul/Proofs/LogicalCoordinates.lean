import bench.repairs.tritonbench_g.int4_matmul.Formula
import bench.repairs.tritonbench_g.int4_matmul.IO
import bench.repairs.tritonbench_g.int4_matmul.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution IO
open scoped BigOperators
noncomputable section

def matrixTarget (s : Shape) (h : s.Valid) (xs : MatrixInputs ℝ s)
    (row : Fin s.m) (col : Fin s.n) : ℝ :=
  ∑ k : Fin s.k, xs.a row k *
    ((weightNibble s h xs k col-zeroNibble s h xs k col : Int):ℝ) *
      xs.scale (groupIndex s h k) col

theorem matrix_bound (m n : Nat) (r : Fin m) (c : Fin n) : r.val*n+c.val<m*n := by
  have h := (finProdFinEquiv (r,c)).isLt
  simpa only [finProdFinEquiv_apply_val,Nat.add_comm,Nat.mul_comm] using h

theorem matrix_read (α : Type) (dtype : TileDType) (m n : Nat)
    (xs : Values α dtype [m*n]) (r : Fin m) (c : Fin n) :
    matrixValues α dtype m n xs r c = xs (⟨r.val*n+c.val,matrix_bound m n r c⟩,()) := by
  change xs (finProdFinEquiv (r,c),()) = _
  congr 1
  apply Prod.ext
  · apply Fin.ext
    simp only [finProdFinEquiv_apply_val]
    ring
  · rfl

theorem sumReal_fin (f : Nat → ℝ) (n : Nat) : sumReal f n = ∑ i : Fin n, f i.val := by
  rw [Fin.sum_univ_eq_sum_range]
  induction n with
  | zero => simp [sumReal]
  | succ n ih => simpa only [sumReal,Finset.sum_range_succ,ih]

theorem matrix_target_eq (s : Shape) (h : s.Valid) (xs : ChannelValues ℝ (inputs s))
    (row : Fin s.m) (col : Fin s.n) :
    matrixTarget s h ((inputCoordinates ℝ s) xs) row col = Formula.mathematical s xs row.val col.val := by
  unfold matrixTarget Formula.mathematical
  rw [sumReal_fin]
  apply Finset.sum_congr rfl
  intro k _
  have ha := matrix_bound s.m s.k row k
  have hw := matrix_bound (s.k/8) s.n (keyPack s h k) col
  have hs := matrix_bound (s.k/s.group) s.n (groupIndex s h k) col
  have hz := matrix_bound (s.k/s.group) (s.n/8) (groupIndex s h k) (columnPack s h col)
  simp only [keyPack,groupIndex,columnPack] at hw hs hz
  change (matrixValues ℝ .real s.m s.k (xs 0) row k) *
    ((nibble (BitVec.ofInt 32 (matrixValues ℝ .int (s.k/8) s.n (xs 1) (keyPack s h k) col)) k.val -
      nibble (BitVec.ofInt 32 (matrixValues ℝ .int (s.k/s.group) (s.n/8) (xs 3)
        (groupIndex s h k) (columnPack s h col))) col.val : Int):ℝ) *
      matrixValues ℝ .real (s.k/s.group) s.n (xs 2) (groupIndex s h k) col = _
  rw [matrix_read ℝ .real s.m s.k (xs 0) row k,
    matrix_read ℝ .int (s.k/8) s.n (xs 1) (keyPack s h k) col,
    matrix_read ℝ .int (s.k/s.group) (s.n/8) (xs 3) (groupIndex s h k) (columnPack s h col),
    matrix_read ℝ .real (s.k/s.group) s.n (xs 2) (groupIndex s h k) col]
  simp only [keyPack,groupIndex,columnPack,Formula.a,Formula.q,Formula.z,Formula.scale,
    dif_pos ha,dif_pos hw,dif_pos hs,dif_pos hz]

/-- Both directions of the whole per-address contract, including exact reset,
all CTA safety/success, locality, coverage, independently ordered outputs/frame. -/
theorem logical_iff (x : Raw.Arguments) (h : x.shape.Valid) (t : Tile)
    (rest : Algebra ℝ) (lanes : List Lane) :
    (logicalIO x t rest lanes).ImplementsOn (matrixDomain x.shape) (matrixTarget x.shape h) ↔
    (programIO x t realOps rest lanes).ImplementsOnCells (domain x.shape realOps)
      (Formula.realOutput x.shape) := by
  have domainEq : (fun raw => matrixDomain x.shape ((inputCoordinates ℝ x.shape) raw)) =
      domain x.shape realOps := by
    funext xs
    simp only [matrixDomain,Equiv.symm_apply_apply]
  have formulaEq : (logicalIO x t rest lanes).lift (matrixTarget x.shape h) = Formula.realOutput x.shape := by
    funext xs o j
    fin_cases o
    exact matrix_target_eq x.shape h xs j.1 j.2.1
  change (programIO x t realOps rest lanes).ImplementsOnCells
    (fun raw => matrixDomain x.shape ((inputCoordinates ℝ x.shape) raw))
    ((logicalIO x t rest lanes).lift (matrixTarget x.shape h)) ↔ _
  rw [domainEq,formulaEq]

theorem logical_real (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (rest : Algebra ℝ)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t)) :
    (logicalIO x t rest lanes).ImplementsOn (matrixDomain x.shape) (matrixTarget x.shape geometry.shape) :=
  (logical_iff x geometry.shape t rest lanes).mpr (real_implements x geometry t member rest lanes complete)
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
