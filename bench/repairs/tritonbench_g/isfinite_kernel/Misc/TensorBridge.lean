import bench.repairs.tritonbench_g.isfinite_kernel.Misc.RealEndpoints
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Kernel
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Storage
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Finite
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.SourceBridge

/-! Optional checked metadata, pointer and physical-byte applications. -/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000

theorem python_checks_iff_preconditions (x y : ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := checkInputs_iff x y

/-- Checked metadata entails physical storage bounds for each byte of every
active input element, and output addresses are injective on active elements. -/
theorem contract_memory_meaning (x y : ViewMetadata) (h : InputContract x y) :
    (∀ i k, i < x.n → k < x.width →
      (x.storageBase : Int) ≤ x.view.signedAddress i+k ∧
      x.view.signedAddress i+k < x.storageBase+x.capacity ∧
      0 ≤ x.view.signedAddress i ∧ x.view.signedAddress i+k < (2^64 : Nat)) ∧
    (∀ i j, i < x.n → j < x.n → y.outputView.address i = y.outputView.address j → i = j) :=
  ⟨fun i k hi hk => accessed_bounds Bool.false x h.input i k hi hk,
    fun i j hi hj => output_injective x y h i j hi hj⟩

/-- The native signed offset and 64-bit pointer expression agree with the
source address for every active lane of a bounded kernel invocation. -/
theorem active_pointer (x y : ViewMetadata) (h : InputContract x y)
    (start count pid lane : Nat) (small : count < 2^31)
    (launched : pid < Finite.grid count) (inTile : lane < 1024)
    (inside : start+(pid*1024+lane) < x.n) :
    Finite.pointer (x.view.shift start) pid lane = x.view.address (start+(pid*1024+lane)) ∧
    Finite.pointer (y.outputView.shift start) pid lane = y.outputView.address (start+(pid*1024+lane)) := by
  refine ⟨Finite.checked_pointer Bool.false x h.input x.n start count pid lane rfl small launched inTile inside,?_⟩
  simpa only [Finite.output_view y h.output] using
    Finite.checked_pointer Bool.true y h.output x.n start count pid lane h.shape small launched inTile inside

/-- Instantiating the source memory from actual raw bytes gives the same
output bytes as the native finite-index load/store program on that heap.
This is one kernel program, with no wrapper execution or allocation claim. -/
theorem kernel_byte_correspondence (x y : ViewMetadata) (h : InputContract x y)
    (start count pid : Nat) (small : count < 2^31)
    (launched : pid < Finite.grid count)
    (inside : ∀ i, i < count → start+i < x.n) (before : Bytes.Heap) :
    ∃ after, FP.Structural.exec (Source.algebra x.precision (Bytes.reference x.precision))
        (source x.precision (x.view.shift start) (y.outputView.shift start) count 1024)
        (Source.backed (Bytes.primitives x.precision.width) before pid) = some after ∧
      (∀ q, Source.cellByte (after.mem "y" q) =
        Lowering.program (Bytes.primitives x.precision.width) (Bytes.reference x.precision)
          (x.view.shift start) (y.outputView.shift start) count pid before q) ∧
      (∀ r q, r ≠ ("y" : RegionName) → after.mem r q =
        (Source.backed (Bytes.primitives x.precision.width) before pid).mem r q) :=
  Source.program_refines (Bytes.primitives x.precision.width) (Bytes.reference x.precision)
    x y h start count pid small launched inside before

theorem isfinite_kernel_correctness (x y : ViewMetadata) (h : InputContract x y)
    (start count B : Nat) (inside : ∀ i, i < count → start+i < x.n)
    (before : State (Bytes.Word x.precision.width)) :
    Spec.Real (∃ after, FP.Structural.exec (Source.algebra x.precision (Bytes.reference x.precision))
        (source x.precision (x.view.shift start) (y.outputView.shift start) count B) before = some after ∧
      Kernel.Post (Bytes.reference x.precision) (x.view.shift start) (y.outputView.shift start)
        count B before after) :=
  RealEndpoints.isfinite_kernel_real x.precision (x.view.shift start) (y.outputView.shift start)
    count B before (Kernel.output_addresses x y h start count inside)

#axiomsClean isfinite_kernel_correctness
#axiomsClean python_checks_iff_preconditions
#axiomsClean contract_memory_meaning
#axiomsClean active_pointer
#axiomsClean kernel_byte_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.TensorBridge
