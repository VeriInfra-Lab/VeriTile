import bench.tritonbench_g.streamk_matmul.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.KernelMemory
open Whole001 Source Checked
noncomputable section

/-- Reuse the established disjoint-store and per-CTA reducer ordering theorem.
It identifies every allowed complete source schedule with the quotation's
canonical execution; no GPU scheduler implementation is proved or assumed. -/
theorem source_schedule (p : Source.Parameters) (ops : Instructions α)
    (initial : Source.State α p exterior)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) :
    Reader.execute kernels p ops initial=some (Schedule.call p ops first reduce last initial) := by
  rw [Reader.execute_projection,Schedule.call_eq]

theorem values_schedule (p : Source.Parameters) (ops : Instructions α) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (initial : Source.State α p exterior) (before : TensorBinding.ValueRead α)
    (binding : ValueBinding v initial before)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) :
    runValues p ops v initial before=some (ValueMemory.call v ops first reduce last (initial,before)) := by
  rw [values_run p ops v laws initial before binding]
  have executed := ValueMemory.call_render v laws ops first reduce last before initial
  rw [initial_values v initial before binding,Schedule.call_eq] at executed
  rw [executed]

theorem bytes_schedule (p : Source.Parameters) (ops : Instructions TensorBinding.Word) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (initial : Source.State TensorBinding.Word p exterior) (before : ByteMemory.Read)
    (binding : ByteBinding v initial before)
    (first : Schedule.Order (firstWave p ops).length) (reduce : Schedule.CTAInterleaving p)
    (last : Schedule.Order (fullWave p ops).length) :
    runBytes p ops v initial before=some (ByteExecution.call v ops first reduce last (initial,before)) := by
  rw [bytes_run p ops v laws initial before binding]
  have executed := ByteExecution.call_render v laws ops first reduce last before initial
  rw [initial_bytes v initial before binding,Schedule.call_eq] at executed
  rw [executed]

#axiomsClean source_schedule
#axiomsClean values_schedule
#axiomsClean bytes_schedule
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.KernelMemory
