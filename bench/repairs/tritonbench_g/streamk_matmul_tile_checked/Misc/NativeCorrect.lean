import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventOriginalDomain
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventCapacity
import VeriTile.Meta.Specification

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

/-! Complete native source observations: every FP32 output word and all four
bytes. The independent ordered K8/atomic target retains the full-tile FP16 round
trip and all FP32 input words. This is a source-to-reference observation, with
the established fixed primitive model, not an arithmetic transformation or
native-to-Real equality; there are no transformation assumptions to report. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.FPEquiv
open Whole001 VeriTile.Triton
noncomputable section

theorem streamk_kernel_native (reference : Native.Reference) (p : Source.Parameters) :
    (EventIO.nativeIO p).Implements (EventIO.nativeArithmetic reference) (EventIO.nativeFormula reference p) :=
  EventIO.native_implements reference p

#stmtSurfaceSubset streamk_kernel_native ⊆
  [Native.Reference,Native.Word,Reader.Sources,EventIO.NativeObservation,Source.Parameters,EventIO.nativeIO,Concurrent.EventProgramIO.Implements,
   EventIO.nativeArithmetic,EventIO.nativeFormula]
#axiomsClean streamk_kernel_native
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.FPEquiv
