import bench.tritonbench_g.streamk_matmul.Proofs.SourceTape
import bench.tritonbench_g.streamk_matmul.Proofs.KernelResults

/-! Exact transport of the same complete shared execution to the established
physical value/byte executors. Initial raw A/B stay immutable logical arrays;
only C and active scratch receive the original physical binding premises. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source Checked KernelMemory VeriTile.Triton
noncomputable section

theorem values_refinement (p : Parameters) (ops : Instructions α) (exterior : Nat)
    (v : ByteExecution.Layout p) (laws : ByteExecution.Laws v)
    (s : Concurrent.State α) (before : Nat → α) (memory : ValueBinding v (extract p exterior s) before) :
    runValues p ops v (extract p exterior s) before = some
      (extract p exterior (Concurrent.run (backend ops) (completeTape p) s),
        ValueMemory.rendered before v (extract p exterior (Concurrent.run (backend ops) (completeTape p) s))) := by
  rw [complete_projection]
  exact values_run p ops v laws _ before memory

theorem bytes_refinement (p : Parameters) (ops : Instructions Native.Word) (exterior : Nat)
    (v : ByteExecution.Layout p) (laws : ByteExecution.Laws v)
    (s : Concurrent.State Native.Word) (before : ByteMemory.Read) (memory : ByteBinding v (extract p exterior s) before) :
    runBytes p ops v (extract p exterior s) before = some
      (extract p exterior (Concurrent.run (backend ops) (completeTape p) s),
        ByteExecution.rendered before v (extract p exterior (Concurrent.run (backend ops) (completeTape p) s))) := by
  rw [complete_projection]
  exact bytes_run p ops v laws _ before memory

#axiomsClean values_refinement
#axiomsClean bytes_refinement
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
