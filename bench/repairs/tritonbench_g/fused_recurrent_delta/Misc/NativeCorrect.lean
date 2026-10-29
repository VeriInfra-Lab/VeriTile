import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IONative
import bench.repairs.tritonbench_g.fused_recurrent_delta.Misc.KernelCompatibility
import VeriTile.Meta.Specification

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry separately proves the actual runtime program pair. -/

/-! Exact output words of the two original complete kernels. Compiler-object
binding, native primitives, reduction routing and the runtime UInt64 scalar
argument remain explicit. These are source observations, with no optimization
candidate, primitive-equals-software assumption or native-equals-Real claim.
There are no floating transformation assumptions for #print_fp_assumptions. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
open VeriTile.Triton FP.Structural KernelProgram KernelNativeCarrier Interface
noncomputable section

theorem fused_recurrent_delta_forward_native (platform : KernelCompiler.Platform)
    (s : Shape) (f : Flags) (binding : KernelCompiler.Binding platform .forward s f)
    (host : UInt64) (p : Program .forward s) :
    (nativeForwardIO platform s f binding host p).Observes (forwardObserver platform s f binding)
      (nativeForwardFormula platform s f binding host p.value) :=
  IONative.forward platform s f binding host p

theorem fused_recurrent_delta_backward_native (platform : KernelCompiler.Platform)
    (s : Shape) (f : Flags) (binding : KernelCompiler.Binding platform .backward s f)
    (host : UInt64) (p : Program .backward s) :
    (nativeBackwardIO platform s f binding host p).Observes (backwardObserver platform s f binding)
      (nativeBackwardFormula platform s f binding host p.value) :=
  IONative.backward platform s f binding host p

#stmtSurfaceSubset fused_recurrent_delta_forward_native ⊆
  [KernelCompiler.Platform,Shape,Flags,KernelCompiler.Binding,Kind.forward,Program,Program.value,
    Node,Word,KernelNativeCarrier.instInhabitedNode,nativeForwardIO,ProgramIO.Observes,forwardObserver,nativeForwardFormula]
#stmtSurfaceSubset fused_recurrent_delta_backward_native ⊆
  [KernelCompiler.Platform,Shape,Flags,KernelCompiler.Binding,Kind.backward,Program,Program.value,
    Node,Word,KernelNativeCarrier.instInhabitedNode,nativeBackwardIO,ProgramIO.Observes,backwardObserver,nativeBackwardFormula]
#axiomsClean fused_recurrent_delta_forward_native
#axiomsClean fused_recurrent_delta_backward_native
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
