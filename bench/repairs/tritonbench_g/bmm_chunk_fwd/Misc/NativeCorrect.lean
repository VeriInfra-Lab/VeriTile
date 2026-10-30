import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeInterface

/-! The source-ordered native formula is an auxiliary correctness result.
The main implementation pair remains in `FPEquiv.lean`. -/
namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph KernelInterface
noncomputable section

/-- Complete native dot recurrence with its casts, K-block/MMA order, full
output readback, source access safety and exterior frame. -/
theorem bmm_chunk_fwd_native (rest : Algebra Expr) (p : Parameters) (chosen : Host.Config) :
    (BmmIO.Native.io rest p chosen).Implements (BmmIO.Native.formula p chosen) :=
  BmmIO.Native.native_implements rest p chosen

/-- Retained combined result from the former reader entry. The two complete
actual schedules are equivalent and the original implements its native formula. -/
theorem bmm_chunk_fwd_execution_and_native
    (rest : Algebra Expr) (p : Parameters) (chosen : Host.Config) :
    (BmmIO.Native.io rest p chosen).toProgramIO.Equivalent
      (BmmIO.Native.candidate rest p chosen) ∧
    (BmmIO.Native.io rest p chosen).Implements (BmmIO.Native.formula p chosen) :=
  ⟨BmmIO.Native.native_equivalent rest p chosen,
    bmm_chunk_fwd_native rest p chosen⟩

#specNonCircular BmmIO.Native.formula avoiding [Kernels.fp32Source,Kernels.indexCommuted,runSchedule,
  FP.Structural.run,FP.Structural.exec,State.mem]
#axiomsClean bmm_chunk_fwd_native
#axiomsClean bmm_chunk_fwd_execution_and_native
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeCorrect

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPEquiv
noncomputable section

/-- Historical native-formula reader alias. -/
abbrev reference := BmmIO.Native.formula

end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPEquiv
