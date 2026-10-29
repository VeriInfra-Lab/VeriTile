import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Host
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Index
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.TensorBinding
open VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost
noncomputable section

variable (x y : VeriTile.Torch.FP32.Tensor) (lx : VeriTile.Torch.FP32.TensorLaws x) (ly : VeriTile.Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata)

def descriptor : Inputs := inputs x.metadata y.metadata
  (matrixBase x y lx h) (vectorBase x y ly h)

/-- Derives the original input/logical-value bridge from shared Torch API laws;
there is no assumed kernel value, reduction or output equation. -/
theorem direct {α : Type} (decode : VeriTile.Torch.FP32.Word → α) (read : Nat → VeriTile.Torch.FP32.Word) :
    DirectBinding α (descriptor x y lx ly h) (memory decode read)
      (matrix x lx decode read) (vector y ly decode read) := by
  constructor
  · intro i j hi hj
    change i < rows x.metadata at hi
    change j < columns x.metadata at hj
    have bound := matrix_index x y h i j hi hj
    have value := congrArg decode (VeriTile.Torch.FP32.direct_read_eq_logical read x lx
      h.contiguous.1 h.noNeg.1 (i*columns x.metadata+j) bound)
    change decode (read (4*(matrixBase x y lx h+i*columns x.metadata+j*1))) = _
    rw [show 4*(matrixBase x y lx h+i*columns x.metadata+j*1) =
        (matrixView x y lx h).address (i*columns x.metadata+j) by
      simp only [VeriTile.Torch.FP32.DataView.address, Nat.mul_one]
      rw [←matrix_base_bytes]
      omega]
    exact value
  · intro j hj
    change j < vectorLength y.metadata at hj
    have bound : j < y.metadata.numel := by
      rw [vector_numel x.metadata y.metadata h, h.dimensions]
      exact hj
    have value := congrArg decode (VeriTile.Torch.FP32.direct_read_eq_logical read y ly
      h.contiguous.2 h.noNeg.2 j bound)
    change decode (read (4*(vectorBase x y ly h+j*1))) = _
    rw [show 4*(vectorBase x y ly h+j*1) = (vectorView x y ly h).address j by
      simp only [VeriTile.Torch.FP32.DataView.address, Nat.mul_one]
      rw [←vector_base_bytes]
      omega]
    exact value

/-- Only output allocation supplies a new span premise. Input address bounds
follow from shared storage bounds, checked contiguity and pointer alignment. -/
theorem bounds {α : Type} {entry : Nat → α} {old : Nat → Prop}
    (a : Allocated α (descriptor x y lx ly h) entry old)
    (outputSpan : 4*a.base + 4*rows x.metadata ≤ 2^64) :
    Index.TensorBounds (descriptor x y lx ly h) a := by
  refine ⟨h.extent.1, h.extent.2, ?_, ?_, ?_, ?_⟩
  · intro v hv
    change v ∈ [columns x.metadata, 1, 1] at hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · exact h.extent.2
    all_goals norm_num
  · intro i j hi hj
    change i < rows x.metadata at hi
    change j < columns x.metadata at hj
    have idx := matrix_index x y h i j hi hj
    have span := (matrixView x y lx h).spanFits
    change (matrixView x y lx h).base + 4*x.metadata.numel ≤ 2^64 at span
    rw [←matrix_base_bytes] at span
    change 4*(matrixBase x y lx h+i*columns x.metadata+j*1)+4 ≤ 2^64
    omega
  · intro j hj
    change j < vectorLength y.metadata at hj
    have idx : j < y.metadata.numel := by
      rw [vector_numel x.metadata y.metadata h, h.dimensions]
      exact hj
    have span := (vectorView x y ly h).spanFits
    change (vectorView x y ly h).base + 4*y.metadata.numel ≤ 2^64 at span
    rw [←vector_base_bytes] at span
    change 4*(vectorBase x y ly h+j*1)+4 ≤ 2^64
    omega
  · intro i hi
    change i < rows x.metadata at hi
    omega

#axiomsClean direct
#axiomsClean bounds
#axiomsClean matrix_pointer
#axiomsClean vector_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.TensorBinding
