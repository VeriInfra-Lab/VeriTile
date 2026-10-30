import bench.tritonbench_g.matmul_dequantize.Proofs.FirstRealResult
import bench.tritonbench_g.matmul_dequantize.Proofs.SecondRealResult
import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant32IOResult
import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant16IOResult
import bench.tritonbench_g.matmul_dequantize.IO
namespace VeriTile.Bench.Repairs.MatmulDequantize.ReaderProof
open VeriTile Triton FP.Structural ReaderIO
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped BigOperators LogicalProgramIO
noncomputable section

theorem coordinate_row (g : Geometry.Layout) (valid : Geometry.Valid g)
    (j : TileIndex [Geometry.count g,g.blockRows,g.blockColumns]) :
    (coordinates g valid j).1.val=Geometry.outRow g j.1.val j.2 := by
  by_cases h : 0<Geometry.gridColumns g
  · simp [coordinates,grid,h,GroupedGrid.mapping,finProdFinEquiv,Geometry.outRow,
      Geometry.blockRow,Nat.add_comm,Nat.mul_comm]
  · have hz : Geometry.gridColumns g=0 := by omega
    have := j.1.isLt
    simp [Geometry.count,hz] at this

theorem coordinate_column (g : Geometry.Layout) (valid : Geometry.Valid g)
    (j : TileIndex [Geometry.count g,g.blockRows,g.blockColumns]) :
    (coordinates g valid j).2.val=Geometry.outColumn g j.1.val j.2 := by
  by_cases h : 0<Geometry.gridColumns g
  · simp [coordinates,grid,h,GroupedGrid.mapping,finProdFinEquiv,Geometry.outColumn,
      Geometry.blockColumn,Nat.add_comm,Nat.mul_comm]
  · have hz : Geometry.gridColumns g=0 := by omega
    have := j.1.isLt
    simp [Geometry.count,hz] at this

theorem matrix_coordinates (A : Type) (g : Geometry.Layout) (valid : Geometry.Valid g)
    (xs : TileIndex [Geometry.count g,g.blockRows,g.blockColumns] → A)
    (j : TileIndex [Geometry.count g,g.blockRows,g.blockColumns]) :
    matrix A g valid xs (coordinates g valid j).1 (coordinates g valid j).2=xs j := by
  change xs ((coordinates g valid).symm (coordinates g valid j))=xs j
  rw [Equiv.symm_apply_apply]

theorem sum_eq (f : Nat → ℝ) (n : Nat) : Real.sum f n=∑ k ∈ Finset.range n,f k := by
  induction n with
  | zero => simp [Real.sum]
  | succ n ih => simp [Real.sum,Finset.sum_range_succ,ih]

theorem gptq_target_eq (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p)) (jobs : List Nat) :
    (ReaderIO.gptq p valid jobs).lift (fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val))=FirstIO.mathematical p := by
  funext xs o j
  fin_cases o
  change (∑ k ∈ Finset.range p.inner, (gptqInputs p xs).arrays.a
      (coordinates (FirstGrid.geometry p) valid j).1.val k *
        (((Real.signedNibble ((gptqInputs p xs).arrays.weight (k/8)
          (coordinates (FirstGrid.geometry p) valid j).2.val) k -
          Real.signedNibble ((gptqInputs p xs).arrays.zero (k/p.groupSize)
            ((coordinates (FirstGrid.geometry p) valid j).2.val/8))
              (coordinates (FirstGrid.geometry p) valid j).2.val : Int) : ℝ) *
          (gptqInputs p xs).arrays.scale (k/p.groupSize) (coordinates (FirstGrid.geometry p) valid j).2.val)) = _
  rw [coordinate_row,coordinate_column]
  simp only [gptqInputs,FirstIO.mathematical,Real.rawMatmul,Real.rawDequant,sum_eq]
  rfl

/-- Equality of the full previous finite typed grid judgment and its logical reader. -/
theorem gptq_judgment_iff (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p)) (jobs : List Nat) :
    (ReaderIO.gptq p valid jobs ⊨ fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val)) ↔
      (FirstIO.realIO p jobs).Implements (FirstIO.mathematical p) := by
  rw [LogicalProgramIO.implements_iff,gptq_target_eq]
  rfl

theorem s2_target_eq (p : FPTypedSecond.Parameters)
    (valid : Geometry.Valid (SecondGrid.geometry p)) (jobs : List Nat) :
    (ReaderIO.s2 p valid jobs).lift (fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val))=SecondIO.mathematical p := by
  funext xs o j
  fin_cases o
  change (∑ k ∈ Finset.range p.inner, (s2Inputs p xs).arrays.a
      (coordinates (SecondGrid.geometry p) valid j).1.val k *
        (((Real.signedNibble ((s2Inputs p xs).arrays.weight (k/8)
          (coordinates (SecondGrid.geometry p) valid j).2.val) k -
          Real.signedNibble ((s2Inputs p xs).arrays.zero (k/p.groupSize)
            ((coordinates (SecondGrid.geometry p) valid j).2.val/8))
              (coordinates (SecondGrid.geometry p) valid j).2.val : Int) : ℝ) *
          (s2Inputs p xs).arrays.scale (k/p.groupSize) (coordinates (SecondGrid.geometry p) valid j).2.val)) = _
  rw [coordinate_row,coordinate_column]
  simp only [s2Inputs,SecondIO.mathematical,Real.rawMatmul,Real.rawDequant,sum_eq]
  rfl

/-- Equality of the full previous finite typed grid judgment and its logical reader. -/
theorem s2_judgment_iff (p : FPTypedSecond.Parameters)
    (valid : Geometry.Valid (SecondGrid.geometry p)) (jobs : List Nat) :
    (ReaderIO.s2 p valid jobs ⊨ fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val)) ↔
      (SecondIO.realIO p jobs).Implements (SecondIO.mathematical p) := by
  rw [LogicalProgramIO.implements_iff,s2_target_eq]
  rfl

theorem d32_target_eq (p : Dequant32.Parameters)
    (valid : Geometry.Valid (DequantGrid32.geometry p)) (jobs : List Nat) :
    (ReaderIO.d32 p valid jobs).lift (fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column))=D32IO.mathematical p := by
  funext xs o j
  fin_cases o
  change (((Real.signedNibble
      (matrix Int _ valid (xs (0 : Fin 3)) (coordinates _ valid j).1 (coordinates _ valid j).2)
        (coordinates _ valid j).1.val -
      Real.signedNibble
        (matrix Int _ valid (xs (1 : Fin 3)) (coordinates _ valid j).1 (coordinates _ valid j).2)
          (coordinates _ valid j).2.val : Int) : ℝ) *
      matrix ℝ _ valid (xs (2 : Fin 3)) (coordinates _ valid j).1 (coordinates _ valid j).2) = _
  rw [matrix_coordinates,matrix_coordinates,matrix_coordinates,coordinate_row,coordinate_column]
  rfl

/-- Equality of the full previous finite typed grid judgment and its logical reader. -/
theorem d32_judgment_iff (p : Dequant32.Parameters)
    (valid : Geometry.Valid (DequantGrid32.geometry p)) (jobs : List Nat) :
    (ReaderIO.d32 p valid jobs ⊨ fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column)) ↔
      (D32IO.realIO p jobs).Implements (D32IO.mathematical p) := by
  rw [LogicalProgramIO.implements_iff,d32_target_eq]
  rfl

theorem d16_target_eq (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters)
    (valid : Geometry.Valid (DequantGrid16.geometry p)) (jobs : List Nat) :
    (ReaderIO.d16 p valid jobs).lift (fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column))=D16IO.mathematical p := by
  funext xs o j
  fin_cases o
  change (((Real.signedNibble
      (matrix Int _ valid (xs (0 : Fin 3)) (coordinates _ valid j).1 (coordinates _ valid j).2)
        (coordinates _ valid j).1.val -
      Real.signedNibble
        (matrix Int _ valid (xs (1 : Fin 3)) (coordinates _ valid j).1 (coordinates _ valid j).2)
          (coordinates _ valid j).2.val : Int) : ℝ) *
      matrix ℝ _ valid (xs (2 : Fin 3)) (coordinates _ valid j).1 (coordinates _ valid j).2) = _
  rw [matrix_coordinates,matrix_coordinates,matrix_coordinates,coordinate_row,coordinate_column]
  rfl

/-- Equality of the full previous finite typed grid judgment and its logical reader. -/
theorem d16_judgment_iff (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters)
    (valid : Geometry.Valid (DequantGrid16.geometry p)) (jobs : List Nat) :
    (ReaderIO.d16 p valid jobs ⊨ fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column)) ↔
      (D16IO.realIO p jobs).Implements (D16IO.mathematical p) := by
  rw [LogicalProgramIO.implements_iff,d16_target_eq]
  rfl

end
end VeriTile.Bench.Repairs.MatmulDequantize.ReaderProof
