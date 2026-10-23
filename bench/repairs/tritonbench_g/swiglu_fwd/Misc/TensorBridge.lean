import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.swiglu_fwd.Correct
import bench.repairs.tritonbench_g.swiglu_fwd.Misc.NativeCorrect
import bench.repairs.tritonbench_g.swiglu_fwd.FPEquiv
import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Binding

/-! Optional applications to the original split/reshape Tensor storage and
Python-check correspondence. Primary kernel specifications do not import this file. -/
namespace VeriTile.Bench.Repairs.SwigluFwd.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Bench.Repairs.SwigluTriton.Native
noncomputable section

/-- Compose the kernel with the original split/reshape storage observations.
X and Y share the input allocation; all storage views use a shared physical
reader. The checked byte-envelope guard proves actual input/output separation.
The existing normalization, when requested by the original Python, is a Torch
API binding. No wrapper, allocation or autotuner execution is proved here. -/
theorem swiglu_tensor_kernel_correct
    (p : Prepared ℝ input supplied entry preexisting) (storage : Storage p)
    (checksPassed : checks p = Bool.true)
    (contract : LaunchContract (metadata p storage))
    (B row tile : Nat) (rowBound : row < p.batchShape.prod) :
    VeriTile.Spec.Real (∃ final,
      exec realAlgebra (kernel "X" "Y" "OUT" p.flat.rowStride p.flat.rowStride
        p.out.rowStride p.ncols B)
        (tileState realAlgebra p.layout B row tile p.state) = some final ∧
      (∀ j : Fin B, tile*B+j.val < p.ncols →
        final.mem "OUT" (row*p.out.rowStride+(tile*B+j.val)) = Cell.mk .real
          (formula (input.read entry (row*(2*p.ncols)+(tile*B+j.val)))
            (input.read entry (row*(2*p.ncols)+(p.ncols+(tile*B+j.val)))))) ∧
      (∀ r o, Outside p.layout B row tile r o → final.mem r o = p.state.mem r o) ∧
      ActiveBytesSeparated p storage ∧ CheckedIndices p storage checksPassed contract) := by
  obtain ⟨final, he, values, frame⟩ := KernelRealResult.swiglu_kernel_correct p.layout B row tile p.state
  refine ⟨final, he, ?_, frame, active_bytes_separated p storage contract,
    checked_indices p storage checksPassed contract⟩
  intro j hj
  have v := values j hj
  obtain ⟨hx, hy⟩ := prepared_reads p row (tile*B+j.val) rowBound hj
  rw [hx, hy] at v
  exact v

/-- Actual storage views initialize one shared physical reader. The complete
native kernel result is expressed in the original tensor's logical halves;
the runtime guard establishes separation of all active input/output bytes. -/
theorem swiglu_tensor_kernel_native (instructions : InstructionSemantics)
    (p : Prepared Word input supplied entry preexisting) (storage : Storage p)
    (checksPassed : checks p = Bool.true)
    (contract : LaunchContract (metadata p storage))
    (B row tile : Nat) (rowBound : row < p.batchShape.prod)
    (scratch : State (Term Word)) :
    ∃ final, exec graphAlgebra (kernel "X" "Y" "OUT" p.flat.rowStride p.flat.rowStride
        p.out.rowStride p.ncols B)
        (tileState graphAlgebra p.layout B row tile
          (KernelNative.rawState ["X", "Y"] (physicalReads p storage) scratch)) = some final ∧
      (∀ j : Fin B, tile*B+j.val < p.ncols →
        interpret (compiledPorts instructions)
          ((final.mem "OUT" (row*p.out.rowStride+(tile*B+j.val))).read .real) =
          fp32Formula instructions (input.read entry (row*(2*p.ncols)+(tile*B+j.val)))
            (input.read entry (row*(2*p.ncols)+(p.ncols+(tile*B+j.val))))) ∧
      (∀ r o, Outside p.layout B row tile r o → final.mem r o =
        (KernelNative.rawState ["X", "Y"] (physicalReads p storage) scratch).mem r o) ∧
      ActiveBytesSeparated p storage ∧ CheckedIndices p storage checksPassed contract := by
  obtain ⟨final, he, values, frame⟩ := KernelNativeResult.swiglu_kernel_native instructions p.layout B row tile
    (physicalReads p storage) scratch
  refine ⟨final, he, ?_, frame, active_bytes_separated p storage contract,
    checked_indices p storage checksPassed contract⟩
  intro j hj
  have value := values j hj
  obtain ⟨hx, hy⟩ := physicalReads_logical p storage row (tile*B+j.val) rowBound hj
  change interpret (compiledPorts instructions)
    ((final.mem "OUT" (row*p.out.rowStride+(tile*B+j.val))).read .real) =
    fp32Formula instructions (physicalReads p storage "X" (row*p.flat.rowStride+(tile*B+j.val)))
      (physicalReads p storage "Y" (row*p.flat.rowStride+(tile*B+j.val))) at value
  rw [hx, hy] at value
  exact value

theorem checked_native_indices (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract (metadata p s)) : CheckedIndices p s hc h :=
  checked_indices p s hc h

theorem checked_physical_inputs [Inhabited α]
    (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    ((p.state.mem "X" (r*p.flat.rowStride+q)).read .real =
      s.preparedRead input.cudaDevice (s.byteBase p.flat.allocation+
        4*(p.flat.base+r*p.flat.rowStride+q))) ∧
    ((p.state.mem "Y" (r*p.flat.rowStride+q)).read .real =
      s.preparedRead input.cudaDevice (s.byteBase p.flat.allocation+
        4*(p.flat.base+r*p.flat.rowStride+(p.ncols+q)))) :=
  prepared_reads_physical p s r q hr hq

theorem original_physical_inputs (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (i : Nat) (hi : i < input.shape.prod) :
    input.read entry i = s.entryRead input.cudaDevice
      (s.byteBase input.allocation+4*input.offset i) :=
  original_read_physical p s i hi

theorem payload_checks_iff (x : VeriTile.Torch.ArgumentMetadata) :
    checkPayload x = Bool.true ↔ ArgumentsAdmitted x := checkPayload_iff x

theorem launch_checks_iff (m : LaunchMetadata) :
    checkLaunch m = Bool.true ↔ LaunchContract m := checkLaunch_iff m

#axiomsClean swiglu_tensor_kernel_correct
#axiomsClean swiglu_tensor_kernel_native
#axiomsClean checked_native_indices
#axiomsClean checked_physical_inputs
#axiomsClean original_physical_inputs
#axiomsClean payload_checks_iff
#axiomsClean launch_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd.TensorBridge
