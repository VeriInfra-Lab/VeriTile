import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelExecution
import VeriTile.Triton.Float.IntegerOutputIO

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IntegerBinding
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- The accepted logical executor keeps signed64 cells; INT8 is the low-byte
observation. This signature preserves that representation exactly. -/
def typedKernel (p : KernelExecution.Parameters) : QuantizeIntegerStage.TypedKernel :=
  ⟨KernelExecution.kernel p,fun _ => none,QuantizeIntegerStage.outputRegister⟩

/-- Exact selected-source connection to the shared llrint executor. This is
not a replacement source, a new primitive law or an assumed run equation. -/
theorem execution_eq (M : Algebra α) (round : α → BitVec 64)
    (equal : α → α → Bool) (p : KernelExecution.Parameters) (s : State α) :
    QuantizeIntegerStage.exec (KernelExecution.backend M equal) (some round) (typedKernel p) s=
      KernelExecution.exec (KernelExecution.backend M equal) round (KernelExecution.kernel p) s := by
  rfl

#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IntegerBinding
