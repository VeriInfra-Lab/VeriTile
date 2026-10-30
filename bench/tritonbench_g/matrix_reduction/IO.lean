import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.matrix_reduction.Proofs.KernelReal
import bench.tritonbench_g.matrix_reduction.Proofs.KernelFP
import VeriTile.Triton.Float.ProgramEquivalence

/-! Public source/precision/channel descriptions shared by the reader entries
and their proof adapters. No execution specification is defined here. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton TypedSource
noncomputable section

/-- Mathematical IO of the selected FP16/FP32 source: a full row-major input
matrix, one output per row, and no scratch. Erasing dtype selects Real
interpretation only; the native source retains its original storage precision. -/
def matrixReductionIO (storage : Storage) (M N : Nat) : MaskedTileKernelIO₁ where
  kernel := (kernel storage "x" "out" N 1 1 M N).eraseDType
  projection := by
    have h : (TypedSource.kernel storage "x" "out" N 1 1 M N).toAlgorithm? =
        .ok (TypedSource.kernel storage "x" "out" N 1 1 M N).toAlgKernel := by
      cases storage <;> rfl
    simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  inp := "x"
  out := "out"
  shape := [M, N]
  read := fun _ i => i.1.val * N + i.2.1.val
  write := fun _ i => i.1.val
  mask := fun _ _ => True
  writeMask := fun _ i => i.2.1.val = 0

end
end VeriTile.Bench.TritonBenchG.MatrixReduction

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect
noncomputable section

/-- Native IO of the authoritative typed source. Storage is exactly FP16 or
FP32 as selected by the source. The primitive backend retains promotion,
reduction order/NaN policy, and destination conversion; all payloads are admitted.
The bound invocation has no program-ID or free scalar-register read; the
three source runtime strides are supplied by its external call bindings. -/
def matrixReductionNativeIO {α : Type} (A : Algebra α) (storage : Storage)
    (M N : Nat) : ProgramIO α α where
  inputs := [⟨(channel storage).toTileDType, [M, N], "x"⟩]
  outputs := [⟨(channel storage).toTileDType, [M], "out"⟩]
  buffers := ["x", "out"]
  read := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.1.val * N + i.2.1.val
  inputMask := fun _ _ _ => True
  write := by
    intro port _
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    exact fun i => i.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := by intro port; exact Fin.elim0 port
  scratchMask := by intro port; exact Fin.elim0 port
  schedule := fun _ => [{
    kernel := kernel storage "x" "out" N 1 1 M N
    backend := A
    pids := fun _ => 0 }]
  encode := id
  observe := fun _ _ => id

/-- Independent ordered native graph, with the source's input promotion,
row maximum primitive, and final narrowing left explicit. -/
def nativeRowMaximum {α : Type} (A : Algebra α) (storage : Storage)
    (M N : Nat) (xs : (matrixReductionNativeIO A storage M N).Inputs) :
    (matrixReductionNativeIO A storage M N).Outputs := by
  intro port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  exact fun i => ofFloat (channel storage) (KernelFP.formula A storage M N
    (fun row col => toFloat (channel storage) (xs (0 : Fin 1) (row, col, ()))) i.1)

end
end VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect
open VeriTile.Triton FP.Structural TypedSource ConcreteCorrect
noncomputable section

/-- Same complete typed public footprint, with direct affine load addressing. -/
def matrixReductionCandidate {α : Type} (A : Algebra α) (storage : Storage)
    (M N : Nat) : ProgramIO.Variant (matrixReductionNativeIO A storage M N) where
  schedule := fun _ => [{
    kernel := candidateKernel storage "x" "out" N 1 1 M N
    backend := A
    pids := fun _ => 0 }]
  observe := fun _ _ => id

end
end VeriTile.Bench.TritonBenchG.MatrixReduction.FPCorrect

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton FP.Structural
noncomputable section

/-- Every original input lane is one ordinary matrix coordinate. -/
def logicalMatrix (M N : Nat) :
    (TileIndex [M,N] → ℝ) ≃ (Fin M → Fin N → ℝ) where
  toFun := fun xs row col => xs (row,col,PUnit.unit)
  invFun := fun xs i => xs i.1 i.2.1
  left_inv := by intro xs; funext i; rcases i with ⟨row,col,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

end
end VeriTile.Bench.TritonBenchG.MatrixReduction

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton FP.Structural
noncomputable section

/-- Logical matrix coordinates; only column zero of each output row is stored,
exactly as in the original judgment. No lane or allocation condition is removed. -/
def rowMaximumIO (storage : TypedSource.Storage) (M N : Nat) :
    LogicalIO (TileIndex [M,N] → ℝ) (TileIndex [M,N] → ℝ)
      (Fin M → Fin N → ℝ) (Fin M → Fin N → ℝ)
      (fun f => (matrixReductionIO storage M N).Implements (fun _ xs => f xs)) where
  inputs := logicalMatrix M N
  outputs := logicalMatrix M N

end
end VeriTile.Bench.TritonBenchG.MatrixReduction
