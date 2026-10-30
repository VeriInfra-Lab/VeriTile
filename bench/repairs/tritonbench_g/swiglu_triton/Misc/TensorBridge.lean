import bench.repairs.tritonbench_g.swiglu_triton.Correct
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.InputBinding

/-! Optional actual-Tensor application and checked native address bindings. -/
namespace VeriTile.Bench.Repairs.SwigluTriton.TensorBridge
open VeriTile.Triton FP.Structural
open Memory DirectHost DirectPhysical
noncomputable section

/-- Kernel output formulas on the admitted caller tensors, using one shared
physical reader and the actual pointer/offset binding. No wrapper theorem. -/
theorem swiglu_tensor_kernel_correct (p : Prepared backward)
    (h : InputBinding.InputContract p) (read : Key → ℝ) (scratch : State ℝ)
    (row : Nat) (rowBound : row < p.rows) :
    VeriTile.Spec.Real (∃ final,
      exec Math.algebra (directSource backward (blockSize p.cols) p.cols)
        (DirectTile.rowState row (InputBinding.initial p read scratch)) = some final ∧
      (∀ ch, enabled backward ch → ∀ col, col < p.cols →
        final.mem (outputRegion ch) (row*p.cols+col) = Cell.mk .real
          (Math.formula ch (read ((p.tensor .a).key (row*p.cols+col)))
            (read ((p.tensor .b).key (row*p.cols+col)))
            (if backward then read ((p.tensor .third).key (row*p.cols+col)) else 0))) ∧
      (∀ r o, DirectTile.Outside backward (blockSize p.cols) p.cols row r o →
        final.mem r o = (InputBinding.initial p read scratch).mem r o)) := by
  obtain ⟨final, he, values, frame⟩ := LogicalExecution.real_kernel backward (blockSize p.cols) p.cols row
    (Nat.le_pow_clog (by decide) p.cols) (InputBinding.initial p read scratch)
  refine ⟨final, he, ?_, frame⟩
  intro ch on col hc
  have inside := index_bound p.rows p.cols row col rowBound hc
  have ha := InputBinding.logical_read p h.1 read scratch .a (row*p.cols+col) inside
  have hb := InputBinding.logical_read p h.1 read scratch .b (row*p.cols+col) inside
  have hd := InputBinding.logical_read p h.1 read scratch .third (row*p.cols+col) inside
  have v := values ch on col hc
  change (InputBinding.initial p read scratch).mem "A" _ = _ at ha
  change (InputBinding.initial p read scratch).mem "B" _ = _ at hb
  rw [ha, hb] at v
  cases backward
  · cases ch <;> simp_all [enabled, Math.formula, DirectSource.region]
  · change (InputBinding.initial p read scratch).mem "DC" _ = _ at hd
    rw [hd] at v
    exact v


theorem python_checks_iff_preconditions (p : Prepared backward) :
    InputBinding.checkInputs p = Bool.true ↔ InputBinding.InputContract p := InputBinding.checkInputs_iff p

theorem checked_native_pointers (p : Prepared backward) (h : Admitted p) (r : Role) (row col : Nat)
    (hr : row < p.rows) (hc : col < p.cols) :
    (BitVec.ofNat 64 (p.tensor r).metadata.dataPtr.toNat +
      BitVec.ofNat 64 (4*(row*p.cols+col))).toNat = ((p.tensor r).key (row*p.cols+col)).address ∧
    ((p.tensor r).key (row*p.cols+col)).address % 4 = 0 := pointer p h r row col hr hc

theorem checked_native_indices (p : Prepared backward) (h : Admitted p) (small : blockSize p.cols ≤ 65536)
    (row lane : Nat) (hr : row < p.rows) (hl : lane < blockSize p.cols) :
    (BitVec.ofNat 64 row * BitVec.ofNat 64 p.cols + BitVec.ofNat 64 lane).toInt =
      (row*p.cols+lane : Nat) := indices p h small row lane hr hl

theorem checked_native_mask (p : Prepared backward) (h : Admitted p) (small : blockSize p.cols ≤ 65536)
    (lane : Nat) (hl : lane < blockSize p.cols) :
    (BitVec.ofNat 32 lane).slt (BitVec.ofNat 32 p.cols) = decide (lane < p.cols) := mask p h small lane hl

#axiomsClean swiglu_tensor_kernel_correct
#axiomsClean python_checks_iff_preconditions
#axiomsClean checked_native_pointers
#axiomsClean checked_native_indices
#axiomsClean checked_native_mask
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.TensorBridge
