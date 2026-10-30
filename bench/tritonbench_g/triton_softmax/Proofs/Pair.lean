import bench.tritonbench_g.triton_softmax.Proofs.NativeIOImplements

namespace VeriTile.Bench.Repairs.TritonSoftmax.Pair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
open scoped VeriTile.Spec
noncomputable section

/-- Reuse both complete source/native proofs on the same input values. Their
independent ordered native row target gives output equality; actual safety,
success and both raw-cell exterior frames remain part of the shared judgment. -/
theorem equivalent (inst : NativeGraph.Instructions) (x y : View) (B p : Nat)
    (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows) :
    IO.original inst x y B p ≡[[]] IO.indexCommuted inst x y B p := by
  apply ProgramIO.Implementation.of_equivalent (IO.indexCandidate inst x y B p)
  apply ProgramIO.EquivalentOn.of_implementsOn
    (NativeIOImplements.correct inst Bool.false x y B p shape injective positive cover row)
  change (IO.nativeIO inst Bool.true x y B p).Implements _
  exact NativeIOImplements.correct inst Bool.true x y B p shape injective positive cover row

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.TritonSoftmax.Pair
