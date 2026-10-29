import bench.repairs.tritonbench_g.masked_select.Correct
import bench.repairs.tritonbench_g.masked_select.Proofs.Contract

/-! Optional checked argument, native-index and prepared-buffer bindings.
The stable mathematical kernel specifications remain in Correct.lean. -/
namespace VeriTile.Bench.Repairs.MaskedSelect.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MaskedSelect
open TypedCopy Whole StableSelection
noncomputable section

theorem python_checks_iff_preconditions (x m : VeriTile.Torch.ArgumentMetadata) (shape : List Nat) :
    checkInputs x m = .ok shape ↔ AdmittedArguments x m shape := checkInputs_iff x m shape

theorem prepared_pointer_check (pointer : Nat) :
    checkPrepared pointer = .ok () ↔ PreparedContract pointer := checkPrepared_iff pointer

/-- Raw argument admission gives the finite extent used by every actual
configuration. Padded lanes retain exact signed-int32 indices and masks;
active accesses are connected to physical descriptors below. -/
theorem checked_native_indices (x m : VeriTile.Torch.TensorMetadata) (shape : List Nat)
    (h : InputContract x m shape) (config : Config) (hc : config ∈ configs)
    (pid lane : Nat) (hp : pid < grid shape.prod config.width) (hl : lane < config.width) :
    (index32 config.width pid lane).toInt = (pid * config.width + lane : Nat) ∧
    mask32 shape.prod config.width pid lane = decide (pid * config.width + lane < shape.prod) :=
  ⟨index32_exact config hc shape.prod pid lane h.extent.2 hp hl,
    mask32_exact config hc shape.prod pid lane h.extent.2 hp hl⟩

/-- Actual physical buffers realize the source byte widths and exact active
addresses. There is no finiteness requirement on FP32 payload words. -/
theorem prepared_memory_meaning (mask : Nat → Bool) (n i pointer : Nat)
    (hn : n < 2^31) (hi : i < n) (buffers : KernelBuffers n (count mask n))
    (binding : pointer = buffers.input.base) :
    PreparedContract pointer ∧
    (buffers.input.address i 0).toNat = pointer + i * 4 ∧
    (buffers.selection.address i 0).toNat = buffers.selection.base + i ∧
    (buffers.cumulative.address i 0).toNat = buffers.cumulative.base + i * 8 ∧
    (mask i = Bool.true →
      (BitVec.ofNat 64 buffers.output.base + destination64 mask i * BitVec.ofNat 64 4).toNat =
        buffers.output.base + count mask i * 4) := by
  subst pointer
  refine ⟨buffers.inputAligned,?_,?_,?_,?_⟩
  · simpa [buffers.inputWidth] using allocation_address buffers.input i 0
      (by simpa [buffers.inputLength] using hi) buffers.input.positive
  · simpa [buffers.selectionWidth] using allocation_address buffers.selection i 0
      (by simpa [buffers.selectionLength] using hi) buffers.selection.positive
  · simpa [buffers.cumulativeWidth] using allocation_address buffers.cumulative i 0
      (by simpa [buffers.cumulativeLength] using hi) buffers.cumulative.positive
  · intro selected
    simpa [buffers.outputWidth] using destination_pointer mask n i hn hi selected
      buffers.output buffers.outputLength 0 buffers.output.positive

#axiomsClean python_checks_iff_preconditions
#axiomsClean prepared_pointer_check
#axiomsClean checked_native_indices
#axiomsClean prepared_memory_meaning
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedSelect.TensorBridge
