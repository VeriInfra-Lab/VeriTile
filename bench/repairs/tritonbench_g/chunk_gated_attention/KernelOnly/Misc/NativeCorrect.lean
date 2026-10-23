import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeObservation

/-! Public native observations of the two selected kernels. NativeIO exposes
the original source, full schedule, finite typed read/write windows and branch
selection. NativeMerge proves equality with the original ordinary-grid result.

ScalarLaws is the retained primitive-realization boundary. There is no kernel
rewrite, finite-operand guard, new numerical law or native-to-Real assertion.
The independent target keeps FP32 loads/casts, the live ordered FMA chain,
zero-plus-seed and exp2 lowering. Arbitrary graph inputs remain admitted. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open NativeInterface
noncomputable section

/-- Every native history and optional final output of the complete recurrent
kernel, with exact stored dtype and exterior cell frame. -/
theorem inner_fp_equiv (c : HostFP.Inner) (actual reference : ScalarNative.Primitives)
    (laws : ScalarLaws actual reference) :
    (innerIO c).Observes (innerObserver c actual) (innerNative c reference) :=
  inner_observes c actual reference laws

/-- Every native prefix output, including the original selected autotune
plans and their guarded dummy invocations, for every original config. -/
theorem prefix_fp_equiv (c : HostFP.Cumulative) (actual reference : ScalarNative.Primitives)
    (laws : ScalarLaws actual reference) :
    (prefixIO c).Observes (prefixObserver c actual) (prefixNative c reference) :=
  prefix_observes c actual reference laws

#axiomsClean inner_fp_equiv
#axiomsClean prefix_fp_equiv
#stmtSurfaceSubset inner_fp_equiv ⊆ [HostFP.Inner,ScalarNative.Primitives,ScalarLaws,
  ProgramIO.Observes,innerIO,innerNative,innerObserver,NativeGraph.Expr,NativeGraph.instInhabitedExpr]
#stmtSurfaceSubset prefix_fp_equiv ⊆ [HostFP.Cumulative,ScalarNative.Primitives,ScalarLaws,
  ProgramIO.Observes,prefixIO,prefixNative,prefixObserver,NativeGraph.Expr,NativeGraph.instInhabitedExpr]
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
