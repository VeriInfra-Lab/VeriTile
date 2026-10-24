import bench.tritonbench_g.mean_reduction.Proofs.SourceTensor
import bench.tritonbench_g.mean_reduction.Proofs.WholeFPValue

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
/-- Flatten original logical coordinates in the exact source permutation order.
The inactive exterior value is irrelevant by bounded-input dependence. -/
def flattenedInput (x : Tensor rank) (dims : List (Fin rank)) (nodup : dims.Nodup)
    (entry : Nat → α) (positive : 0<columns x.shape dims) (i : Nat) : α :=
  if hi : i<rows x.shape dims*columns x.shape dims then
    compressedValue x.shape dims nodup x.stride (x.read entry)
      ⟨i/columns x.shape dims,(Nat.div_lt_iff_lt_mul positive).mpr hi⟩
      ⟨i%columns x.shape dims,Nat.mod_lt _ positive⟩
  else default

/-- Independent primitive-operation result on original tensor coordinates.
No runtime, allocator, kernel execution, or computed output is an argument. -/
def fpFormula (A : Algebra α) (x : Tensor rank) (dims : List (Fin rank)) (nodup : dims.Nodup)
    (entry : Nat → α) (positive : 0<columns x.shape dims) (row : Nat) : α :=
  WholeFPValue.rowValue A "X" (rows x.shape dims) (columns x.shape dims)
    (fun _ i => Cell.mk .real (flattenedInput x dims nodup entry positive i)) row


end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
