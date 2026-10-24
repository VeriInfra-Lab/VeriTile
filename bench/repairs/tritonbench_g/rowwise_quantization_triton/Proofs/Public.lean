import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Host
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Mathematics
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false

/-- Complete observable result of the actual host composition. The right-hand
sides are scalar expressions over the original logical input, not API-supplied
kernel results. Byte equality includes implicit int8 narrowing and both bytes
of the direct binary16 store. -/
structure ResultCorrect (ops : Operations Word) (p : Launch)
    (input : Input p.rows p.columns) (before : State) (q m : Tensor) (after : Heap) : Prop where
  quantized_type : q.dtype = .int8
  quantized_shape : q.shape = [p.rows, p.columns]
  maximum_type : m.dtype = .float16
  maximum_shape : m.shape = [p.rows]
  quantized : ∀ r, r < p.rows → ∀ c, c < p.columns →
    after (q.base + r * p.columns + c) = rowQuantized ops (input.logical r c)
      (rowMaximum ops p (input.logical r))
  maximum : ∀ r, r < p.rows → ∀ byte : Fin 2,
    after (m.base + 2 * r + byte.val) = halfByte
      (ops.toHalf (rowMaximum ops p (input.logical r))) byte
  caller_frame : ∀ address, before.live address → after address = before.heap address

theorem Correct (api : Services) (ops : Operations Word) (p : Launch) (schedule : Schedule p)
    (input : Input p.rows p.columns) (before : State) (q m : Tensor) (after : Heap)
    (run : forward api ops p schedule input before = some (q, m, after)) :
    ResultCorrect ops p input before q m after := by
  obtain ⟨hq, hqs, hm, hms, qv, mv, frame⟩ := returned api ops p schedule input before q m after run
  exact ⟨hq, hqs, hm, hms, qv, mv, frame⟩

/-- Active flattened indices fit the widened source arithmetic whenever the
ordinary backing element count fits signed64. Inactive lanes perform no access;
native launch/grid realization remains an ordinary runtime boundary. -/
theorem active_index_int64 (p : Launch) (r c : Nat) (hr : r < p.rows)
    (hc : c < p.columns) (backing : p.rows * p.columns < 2 ^ 63) :
    r < 2 ^ 63 ∧ p.columns < 2 ^ 63 ∧ r * p.columns < 2 ^ 63 ∧
      r * p.columns + c < 2 ^ 63 := by
  have total := linear_bound p.rows p.columns r c hr hc
  have rowbound : r ≤ r * p.columns := by
    simpa using Nat.mul_le_mul_left r p.positive
  have colbound : p.columns ≤ p.rows * p.columns := by
    simpa using Nat.mul_le_mul_right p.columns (show 1 ≤ p.rows by omega)
  omega

/-- Empty batches return fresh tensors and execute no source stores. -/
theorem empty_grid (ops : Operations Word) (p : Launch) (v : Layout)
    (schedule : Schedule p) (heap : Heap) (empty : p.rows = 0) :
    runGrid ops p v schedule.order heap = heap := by
  have hs : schedule.order = [] := by simpa [empty] using schedule.complete
  simp [runGrid, hs]

#axiomsClean Correct
#axiomsClean active_index_int64
#axiomsClean empty_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
