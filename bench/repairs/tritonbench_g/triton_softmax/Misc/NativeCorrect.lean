import bench.repairs.tritonbench_g.triton_softmax.IO
import bench.repairs.tritonbench_g.triton_softmax.Proofs.NativeIOImplements

namespace VeriTile.Bench.Repairs.TritonSoftmax.NativeCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
noncomputable section

/-- Selected source/precision/ABI and explicit channel windows are in `IO.lean`. -/
abbrev nativeIO := IO.nativeIO

/-- Independent ordered native reduction and scalar instruction graph. The
instruction ports describe the fixed primitive model, without numerical rewrites. -/
def reference (inst : NativeGraph.Instructions) (n B : Nat)
    (xs : Fin B → NativeGraph.Word) (i : Fin B) : NativeGraph.Word :=
  NativeGraph.row inst 1 n B (fun j => if hj : j<B then xs ⟨j,hj⟩ else 0) i.val

/-- Both actual FP32 sources execute safely, return the independent native row
and preserve every exterior typed cell. The primitive model admits all words. -/
theorem kernel_native (inst : NativeGraph.Instructions) (candidate : Bool) (x y : View)
    (B p : Nat) (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows) :
    (nativeIO inst candidate x y B p).Implements
      (IO.nativeValues inst candidate x y B p (reference inst x.cols B)) :=
  NativeIOImplements.correct inst candidate x y B p shape injective positive cover row

#specNonCircular reference avoiding [softmax_kernel,indexCommutedKernel,Floating.evaluate]
#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [nativeIO,reference,IO.nativeValues,ProgramIO.Implements,
   NativeGraph.Instructions,NativeGraph.Node,NativeGraph.Word,SoftmaxTriton1.NativeGraph.Word,
   RawKernel.View,RawKernel.View.rows,RawKernel.View.cols,RawKernel.View.Injective]
end
end VeriTile.Bench.Repairs.TritonSoftmax.NativeCorrect
