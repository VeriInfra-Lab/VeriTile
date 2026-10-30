import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.IOImplements
import bench.repairs.tritonbench_g.spinning_lock_reduction.Misc.FPStageConnection

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.FPEquiv
open VeriTile VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

/-- Native observation of both source stages in the existing FP32 model.
The finite boundary retains every FP32 word allowed by the original logical
publication conditions. `+0 + +0 = +0` is proved in that model; it is neither
an experimental assumption nor an additional floating transformation. -/
theorem spinning_kernel_native (h : HostArgs) (buf : Wide.Buffers h)
    (stage : Phased.Stage) :
    (IO.io (FP.Value .fp32) h buf.cElements stage).ImplementsOn (ConcurrentSource.backend fp32)
      (IO.domain fp32.zero h buf.cElements stage) (IO.mathematical fp32.zero h buf.cElements stage) :=
  IOProof.implements fp32 fp32_positive_zero_add h buf stage

#stmtSurfaceSubset spinning_kernel_native ⊆
  [IO.io,IO.domain,IO.mathematical,Concurrent.EventProgramIO.ImplementsOn,
   ConcurrentSource.backend,fp32,Backend.zero,FP.Value,FP.Format.fp32,
   HostArgs,Wide.Buffers,Wide.Buffers.cElements,Phased.Stage,SourceReader.Statement]
#specNonCircular Formula.stage avoiding
  [runKernel,SourceReader.runPlan,Concurrent.Controller.run,Concurrent.execute,IO.io]
#specNonCircular Formula.Domain avoiding
  [runKernel,SourceReader.runPlan,Concurrent.Controller.run,
   Concurrent.EventProgramIO.ImplementsOn,Formula.stage]
#axiomsClean spinning_kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.FPEquiv
