import bench.repairs.tritonbench_g.rope_backward_transform.Correct
import bench.repairs.tritonbench_g.rope_backward_transform.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelMath
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelAccess

/- The bridge retains the complete admitted Tensor applications and byte heap.
Raw GridExec and Tensor Execution use the same row_execution certificate;
KernelContract.source_binding identifies the actual JIT with that source.
scalar_binding/real_binding/native_binding supply the explicit correspondence
from raw initial-memory results to the Tensor logical coordinates. -/
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.TensorBridge
open VeriTile.Triton FP.Structural
open KernelContract
noncomputable section

/-- Kernel-only fixed-coefficient input adjoint for both actual prepared Tensor
buffers. The complete row grid executes the actual typed JIT; all Q/K logical
coordinates and the regional/physical frame follow from that same execution. -/
theorem rope_backward_kernel_correct (read : Nat→Word) (interpret : Word→ℝ)
    (c : Call) (admit : Admitted c) :
    InputBinding c read ∧ Access c ∧
    ∃ final,Execution ideal.backend c (launchRows c.geometry) (memory id (interpret ∘ read) c) final ∧
      (∀ key row head col,row<c.geometry.rows →head<c.geometry.heads key →col<c.geometry.dimension →
        logicalRead (physical id c (interpret ∘ read) final) (c.q key) (c.geometry.offset key row head col)=
          realValue c (interpret ∘ read) key row head col) ∧
      (∀ name offset,¬WrittenRegion c.geometry name offset →final name offset=memory id (interpret ∘ read) c name offset) ∧
      (∀ address,¬WrittenWord c Bool.false address ∧ ¬WrittenWord c Bool.true address →
        physical id c (interpret ∘ read) final address=(interpret ∘ read) address) := by
  refine ⟨input_binding c admit read,access c admit,
    grid ideal.backend c.geometry (memory id (interpret ∘ read) c) (launchRows c.geometry),
    grid_success ideal.backend c _ _,?_,grid_exterior ideal id c (interpret ∘ read),physical_frame id c _ _⟩
  intro key row head col hr hh hc
  simpa only [real_binding,id_eq] using
    physical_values ideal id id (fun _=>rfl) c admit (interpret ∘ read) key row head col hr hh hc

/-- Empty rows, dimensions below two, and two empty head sets submit no rows.
The kernel grid relation returns the same entry memory in these cases. -/
theorem rope_backward_kernel_no_launch {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (m : Source.Memory α)
    (off : c.geometry.rows=0 ∨ c.geometry.dimension<2 ∨ (c.geometry.qHeads=0 ∧ c.geometry.kHeads=0)) :
    launchRows c.geometry=[] ∧ Execution M c (launchRows c.geometry) m m := by
  have noton : ¬launched c.geometry := by
    intro on
    rcases off with h | h | ⟨q,k⟩
    · have rows:=on.1; omega
    · have half:=on.2.1; unfold Request.half at half; omega
    · rcases on.2.2 with h | h <;> omega
  have empty : launchRows c.geometry=[] := by simp [launchRows,noton]
  exact ⟨empty,empty ▸ Execution.nil m⟩

#axiomsClean rope_backward_kernel_correct
#axiomsClean rope_backward_kernel_no_launch
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.TensorBridge

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.TensorBridge
open VeriTile.Triton FP.Structural
open KernelContract
noncomputable section

/-- Actual Tensor and byte-heap inputs, complete typed grid success, every Q/K
native output word read from one completed byte heap, all four output bytes,
and the complete regional and physical byte frame are combined here. The
primitive FP32 cfg and observed coefficient-first FMA lowering are the model
boundary; no numerical exactness or successful execution is a premise. -/
theorem rope_backward_kernel_fp32 (cfg : FP.Config) (c : Call) (admit : Admitted c) (before : Bytes) :
    InputBinding c (read32 before) ∧ Access c ∧
    ∃ final,Execution (Numerics.operations (nativePrimitives cfg)).backend c (launchRows c.geometry)
        (memory Numerics.leaf (read32 before) c) final ∧
      (∀ key row head col,row<c.geometry.rows →head<c.geometry.heads key →col<c.geometry.dimension →
        logicalRead (read32 (byteMemory c before final)) (c.q key) (c.geometry.offset key row head col)=
          nativeValue (nativePrimitives cfg) c (read32 before) key row head col) ∧
      (∀ key row head col,row<c.geometry.rows →head<c.geometry.heads key →col<c.geometry.dimension →
        ∀ byte : Fin 4,byteMemory c before final
            ((c.q key).storage.address ((c.q key).physicalIndex (c.geometry.offset key row head col))+byte.val)=
          wordByte (nativeValue (nativePrimitives cfg) c (read32 before) key row head col) byte) ∧
      (∀ name offset,¬WrittenRegion c.geometry name offset →
        final name offset=memory Numerics.leaf (read32 before) c name offset) ∧
      (∀ address,¬WrittenByte c Bool.false address ∧ ¬WrittenByte c Bool.true address →
        byteMemory c before final address=before address) := by
  refine ⟨input_binding c admit (read32 before),access c admit,
    grid (Numerics.operations (nativePrimitives cfg)).backend c.geometry
      (memory Numerics.leaf (read32 before) c) (launchRows c.geometry),
    grid_success _ c _ _,?_,?_,grid_exterior _ Numerics.leaf c (read32 before),bytes_frame c before _⟩
  · exact native_words (nativePrimitives cfg) c admit before
  · exact native_bytes (nativePrimitives cfg) c admit before

#axiomsClean rope_backward_kernel_fp32
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.TensorBridge

