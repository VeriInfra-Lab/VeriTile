import bench.repairs.tritonbench_g.decay_cumsum.IO
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.DecayCumsum.FPEquiv
open VeriTile Triton FP.Structural Geometry
open scoped VeriTile.Spec
noncomputable section

/-- Complete source-to-native observation under the selected object's original
runtime FP32 scale ABI. All input Nodes/words, three kernel branches, complete
main/tail streams, in-place outputs and exterior cells retain their prior
scope. FMA/order/FTZ-exp2 are explicit primitive model data; no floating
transformation assumption or Real/native equality is introduced. -/
theorem decay_cumsum_kernel_equiv (platform : KernelCompiler.Platform)
    (kind : Kind) (h : Call) (p : Phase) (binding : KernelCompiler.Binding platform kind h p)
    (host : UInt64) (hp : p∈phases h) (prog : Program h p) :
    (DecayIO.io (KernelCompiler.algebra platform binding host) kind h p prog Native.scaleSlot).Observes
      (DecayIO.nativeObservation kind h p)
      (DecayIO.nativeTarget (platform.instructions binding.object.primitives) kind h p
        (platform.scale binding.object host)) :=
  IOCorrect.compiled platform kind h p binding host hp prog

#specNonCircular DecayIO.nativeTarget avoiding [Kernels.source,FP.Structural.exec,FPLogical.evaluate]
#axiomsClean decay_cumsum_kernel_equiv
#stmtSurfaceSubset decay_cumsum_kernel_equiv ⊆
  [ProgramIO.Observes, DecayIO.io, DecayIO.nativeObservation, DecayIO.nativeTarget,
   KernelCompiler.Platform, KernelCompiler.Binding, KernelCompiler.algebra,
   KernelCompiler.Platform.instructions, KernelCompiler.Binding.object,
   KernelCompiler.Object.primitives, KernelCompiler.Platform.scale,
   Kind, Call, Phase, phases, Program, Native.scaleSlot, Native.Node, Native.Word,
   Native.instInhabitedNode]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.FPEquiv
