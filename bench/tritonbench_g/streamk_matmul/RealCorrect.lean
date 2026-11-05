import bench.tritonbench_g.streamk_matmul.Proofs.LogicalIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Correct
open Whole001 Source VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Full M×N matrix product over exactly K terms. Complete A/B tensors and
arbitrary initial C/scratch snapshots retain all original input coordinates.
The full decoded source, concurrent execution, safety and frame are unchanged. -/
specification streamk_kernel_real (p : Source.Parameters) :
    Spec.Real (MatrixIO.io p ⊨ fun xs row col =>
      ∑ k : Fin p.h.k, xs.a row k * xs.b k col) :=
  (MatrixIO.judgment_iff p).2 (EventIO.real_implements p)

#axiomsClean streamk_kernel_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Correct
