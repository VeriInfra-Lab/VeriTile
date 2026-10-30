import bench.repairs.tritonbench_g.matrix_vector_multip.Kernels
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Contract
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeContract

namespace VeriTile.Bench.Repairs.MatrixVectorMultip
open VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost
noncomputable section

def inputs (x y : TensorMetadata) (matrixBase vectorBase : Nat) : Inputs where
  rows := rows x
  columns := columns x
  vectorLength := vectorLength y
  matrixBase := matrixBase
  vectorBase := vectorBase
  strideRow := columns x
  strideColumn := 1
  strideVector := 1
  device := x.device.index

/-- Ordinary input-only binding of the two original logical tensors to their
actual aligned data pointers. Inputs are never copied or modified. The FP32
instance is derived from shared Torch laws in Proofs/TensorBinding. -/
structure DirectBinding (α : Type) (x : Inputs) (entry : Nat → α)
    (matrix : Nat → Nat → α) (vector : Nat → α) : Prop where
  matrixCoordinate : ∀ i j, i < x.rows → j < x.columns →
    entry (x.matrixAddress i j) = matrix i j
  vectorCoordinate : ∀ j, j < x.vectorLength → entry (x.vectorAddress j) = vector j

theorem compatible (x y : TensorMetadata) (h : InputContract x y) (ma vb : Nat) :
    (inputs x y ma vb).columns = (inputs x y ma vb).vectorLength := h.dimensions

/-- Historical natural-number host model only. The actual raw while source
is connected by SourceOutput.tile_result and TensorBridge. -/
theorem historical_source_binding (x y : TensorMetadata) (ma vb : Nat) (c : Config) :
    WholeSource.source ((inputs x y ma vb).parameters c) =
      Typed.kernel .fp32 ((inputs x y ma vb).parameters c) := rfl

/-- The wrapper checks metadata before allocation or launch. -/
def executeChecked {α : Type} {x y : TensorMetadata} {ma vb : Nat}
    {entry : Nat → α} {old : Nat → Prop}
    (run : WholeHost.Runtime α (inputs x y ma vb) entry old → HostOutcome α)
    (rt : WholeHost.Runtime α (inputs x y ma vb) entry old) :
    Except InputError (HostOutcome α) := do
  checkTensorInputs x y
  pure (run rt)

theorem checked_run {α : Type} {x y : TensorMetadata} {ma vb : Nat}
    {entry : Nat → α} {old : Nat → Prop}
    (h : InputContract x y)
    (run : WholeHost.Runtime α (inputs x y ma vb) entry old → HostOutcome α)
    (rt : WholeHost.Runtime α (inputs x y ma vb) entry old) :
    executeChecked run rt = .ok (run rt) := by
  simp [executeChecked, (checkTensorInputs_iff x y).mpr h]

#axiomsClean historical_source_binding
#axiomsClean checked_run
end
end VeriTile.Bench.Repairs.MatrixVectorMultip
