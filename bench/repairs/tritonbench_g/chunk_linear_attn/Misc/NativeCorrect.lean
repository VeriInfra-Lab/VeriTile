import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.PublicNativeIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.FPEquiv
open VeriTile Triton FP.Structural
noncomputable section

/-- Same four complete quotations and finite FP32 reduction. All six result
and three intermediate windows are observable, with their exact native words. -/
abbrev io (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (scale : BitVec 32) (work : KernelSchedule.Work) := IO.native profile c scale work

/-- Source-to-native FP32/FMA observation with the actual scale ABI word,
every legal per-stage grid order and original finite reduction plans. The
primitive instruction interpretation is a model boundary; no arithmetic
rewrite or equality between native and Real arithmetic is assumed. -/
theorem kernel_native (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (valid : Host.Valid c.dims) (scale : BitVec 32) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.call work) :
    (io profile c scale work).Implements (NativeFormula.native profile c valid scale) :=
  PublicNativeIO.native profile c valid scale work legal

#stmtSurfaceSubset kernel_native ⊆
  [NativeTerms.FP32Profile, KernelInterface.Config, KernelInterface.Config.dims,
    Host.Valid, KernelSchedule.Work, KernelSchedule.Legal, KernelInterface.Config.call,
    io, ReductionProgramIO.Implements, NativeFormula.native, NativeTerms.Term, NativeTerms.instInhabitedTerm]
#specNonCircular NativeFormula.native avoiding [KernelSchedule.run,KernelExecution.run,
  KernelExecution.runNative, FP.Structural.exec,runGrid,executeFour,gridMemory,
  stageMemory,modelFour,completeMemory,NativeTerms.rawMemory,NativeBound.rawMemory]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.FPEquiv
