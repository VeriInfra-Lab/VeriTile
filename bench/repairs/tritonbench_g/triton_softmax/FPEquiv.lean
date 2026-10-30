import bench.repairs.tritonbench_g.triton_softmax.Proofs.Pair

namespace VeriTile.Bench.Repairs.TritonSoftmax.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
open scoped VeriTile.Spec
noncomputable section

abbrev original := IO.original
abbrev indexCommuted := IO.indexCommuted

/-- Compare both actual complete FP32 row invocations under the retained
fixed-four-warp native graph. Only integer input-row multiplication is commuted;
max/sub/exp/sum/div order, every output-thread observation, source safety,
successful execution and both complete exterior cell frames are preserved. -/
specification kernel_native (inst : NativeGraph.Instructions) (x y : View)
    (B p : Nat) (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows) :
    original inst x y B p ≡[[]] indexCommuted inst x y B p :=
  Pair.equivalent inst x y B p shape injective positive cover row

#stmtSurfaceSubset kernel_native ⊆
  [original,indexCommuted,Spec.Equiv,Spec.RuleEntry,Spec.ProgramSyntax.Statement,
   ProgramIO.Inputs,ProgramIO.InputBindingMode.decoded,ProgramIO.Implementation,
   ProgramIO.Implementation.programSyntax,IO.nativeIO,NativeGraph.Instructions,
   NativeGraph.Node,NativeGraph.Word,SoftmaxTriton1.NativeGraph.Word,
   RawKernel.View,RawKernel.View.rows,RawKernel.View.cols,RawKernel.View.Injective]
#print_fp_assumptions kernel_native
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonSoftmax.FPEquiv
