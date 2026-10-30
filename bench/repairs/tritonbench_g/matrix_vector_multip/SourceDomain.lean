import bench.repairs.tritonbench_g.matrix_vector_multip.Contract
import VeriTile.Triton.DSL.RandomProfile
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.MatrixVectorMultip
open VeriTile.Triton

/-- Conditions for projecting the actual uint64 source onto its mathematical
row/column coordinates. These come from legal launch arguments and active
backing addresses; no accumulator or output equality is assumed. -/
structure SourceDomain (N M BN BM pid : Nat) : Prop where
  rows : N < 2^63
  columns : M < 2^63
  program : pid < 2^31
  matrixOffsets : ∀ row col, row < N → col < M → row*M+col < 2^64

namespace SourceDomain

theorem row_exact_bound (h : SourceDomain N M BN BM pid)
    (shape : Random.ArangeSizeDomain BN) (lane : Fin BN) :
    pid*BN+lane.val < 2^64 := by
  have blockBound := shape.2.1
  have programBound := h.program
  have laneBound := lane.isLt
  have productBound := Nat.mul_le_mul (Nat.le_of_lt programBound) blockBound
  omega

theorem final_counter_bound (h : SourceDomain N M BN BM pid)
    (shape : Random.ArangeSizeDomain BM) :
    ((M+BM-1)/BM)*BM < 2^64 := by
  have chunkBound := Nat.div_mul_le_self (M+BM-1) BM
  have blockBound := shape.2.1
  have columnBound := h.columns
  omega

theorem active_column_bound (h : SourceDomain N M BN BM pid)
    (shape : Random.ArangeSizeDomain BM) (off : Nat) (active : off < M)
    (lane : Fin BM) : off+lane.val < 2^64 := by
  have blockBound := shape.2.1
  have columnBound := h.columns
  have laneBound := lane.isLt
  omega

/-- Every actual checked Tensor invocation discharges the new source domain.
The old mathematical theorem had no such machine-integer conditions. -/
theorem of_kernel_contract {x y : VeriTile.Torch.TensorMetadata}
    {c : KernelConfig} {pid : Nat} {memory : KernelMemory (MatrixVectorMultip.rows x) (MatrixVectorMultip.columns x)}
    (contract : KernelContract x y c pid memory) :
    SourceDomain (MatrixVectorMultip.rows x) (MatrixVectorMultip.columns x) c.blockRows c.blockColumns pid := by
  refine ⟨contract.inputs.extent.1, contract.inputs.extent.2, contract.programABI, ?_⟩
  intro row col hr hc
  have bound := memory.matrixBound row col hr hc
  omega

end SourceDomain
end VeriTile.Bench.Repairs.MatrixVectorMultip
