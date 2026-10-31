import bench.tritonbench_g.matrix_vector_multip.Contract
import bench.tritonbench_g.matrix_vector_multip.Sources.Parameters
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip
open VeriTile.Triton
/-- Input metadata is signed-64 and the constexpr blocks fit its positive range.
These are source conversion/control bounds, not output assumptions. -/
structure SourceBounds (N M BN BM : Nat) : Prop where
  rows : N < 2^63
  columns : M < 2^63
  blockRows : BN ≤ 2^63
  blockColumns : BM ≤ 2^63
namespace SourceBounds

theorem scalar_rows (h : SourceBounds N M BN BM) : N<2^64 := by have := h.rows; omega
theorem scalar_columns (h : SourceBounds N M BN BM) : M<2^64 := by have := h.columns; omega
theorem lanes_rows (h : SourceBounds N M BN BM) : BN≤2^64 := by have := h.blockRows; omega
theorem lanes_columns (h : SourceBounds N M BN BM) : BM≤2^64 := by have := h.blockColumns; omega

theorem counter_no_wrap (h : SourceBounds N M BN BM) (off : Nat) (active : off<M) :
    (off+BM) % 2^64=off+BM := by
  apply Nat.mod_eq_of_lt
  have := h.columns; have := h.blockColumns; omega

theorem grid_pid (h : SourceBounds N M BN BM) (pid : Nat)
    (legal : pid<(N+BN-1)/BN) : pid<2^64 := by
  have bound := Nat.div_le_self (N+BN-1) BN
  have := h.rows; have := h.blockRows
  omega

theorem grid_row_no_wrap (h : SourceBounds N M BN BM) (positive : 0<BN)
    (pid lane : Nat) (legal : pid<(N+BN-1)/BN) (inside : lane<BN) :
    (pid*BN+lane)%2^64=pid*BN+lane := by
  have hp := Nat.mul_le_mul_right BN (Nat.succ_le_iff.mpr legal)
  simp only [Nat.succ_mul] at hp
  have hd := Nat.div_mul_le_self (N+BN-1) BN
  have := h.rows; have := h.blockRows
  apply Nat.mod_eq_of_lt
  omega

theorem of_input_config (x y : VeriTile.Torch.TensorMetadata) (c : KernelConfig)
    (inputs : InputContract x y) : SourceBounds (MatrixVectorMultip.rows x) (MatrixVectorMultip.columns x) c.blockRows c.blockColumns := by
  refine ⟨inputs.extent.1,inputs.extent.2,?_,?_⟩
  · have := c.rows_mem
    simp only [List.mem_cons,List.not_mem_nil,or_false] at this
    rcases this with h|h|h|h <;> omega
  · have := c.columns_mem
    simp only [List.mem_cons,List.not_mem_nil,or_false] at this
    rcases this with h|h|h <;> omega
#axiomsClean counter_no_wrap
#axiomsClean grid_row_no_wrap
#axiomsClean of_input_config
end SourceBounds
end VeriTile.Bench.Repairs.MatrixVectorMultip
