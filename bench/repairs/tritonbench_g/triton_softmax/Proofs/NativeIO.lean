import bench.repairs.tritonbench_g.triton_softmax.Proofs.KernelInterface
import bench.repairs.tritonbench_g.triton_softmax.Proofs.StructuralAccess
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.TritonSoftmax.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Independent fixed-four-warp native row, padding outside the source tile. -/
def reference (inst : NativeGraph.Instructions) (n B : Nat)
    (xs : Fin B → NativeGraph.Word) (i : Fin B) : NativeGraph.Word :=
  NativeGraph.row inst 1 n B (fun j => if hj : j<B then xs ⟨j,hj⟩ else 0) i.val

/-- The native observation is read from the actual source's stored tagged cell. -/
theorem result (inst : NativeGraph.Instructions) (x y : View) (B p : Nat)
    (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows)
    (s : State NativeGraph.Node) (xs : Fin B → NativeGraph.Word)
    (input : ∀ j : Fin B,j.val<x.cols →
      (s.mem x.region (x.index p j.val)).read .real=NativeGraph.atom (xs j)) :
    ∃ t,Floating.evaluate (NativeGraph.algebra inst 1) x.descriptor y.descriptor B p s=some t ∧
      (∀ i : Fin B,i.val<x.cols →
        mapCell (fun value => value (NativeGraph.owner 1 B i.val))
          (t.mem y.region (y.index p i.val)) = .mk .real (reference inst x.cols B xs i)) ∧
      (∀ r off,(∀ i : Fin B,i.val<x.cols → ¬(y.region=r ∧ y.index p i.val=off)) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht⟩ := Floating.terminates (NativeGraph.algebra inst 1) x.descriptor y.descriptor B p positive s
  obtain ⟨u,hu,hm⟩ := Floating.source_memory (NativeGraph.algebra inst 1) x.descriptor y.descriptor B p positive s t ht
  have inj : ∀ i j : Fin B,i.val<x.cols → j.val<x.cols → i.val*y.colStride=j.val*y.colStride → i=j := by
    intro i j hi hj eq
    apply Fin.ext
    exact (injective p i.val p j.val (by rwa [shape.1]) (by rwa [shape.2])
      (by rwa [shape.1]) (by rwa [shape.2]) (by simp [View.index,eq])).2
  refine ⟨t,ht,?_,?_⟩
  · intro i hi
    have value := SoftmaxTriton1.Floating.row_value (NativeGraph.algebra inst 1)
      y.region x.region x.offset y.offset x.rowStride x.colStride y.rowStride y.colStride
      x.cols B positive inj (Floating.launchState x.descriptor p s) u hu i hi
    have inp : ∀ j : Fin B,j.val<x.cols →
        ((Floating.launchState x.descriptor p s).mem x.region
          (x.offset+(Floating.launchState x.descriptor p s).pids 0*x.rowStride+j.val*x.colStride)).read .real=
          NativeGraph.atom (xs j) := input
    rw [SoftmaxTriton1.Floating.expression_congr (NativeGraph.algebra inst 1) x.cols B _ _ inp i] at value
    change u.mem y.region (y.index p i.val)=_ at value
    rw [←hm,value]
    change (Cell.mk .real ((SoftmaxTriton1.Floating.rowExpression (NativeGraph.algebra inst 1)
      x.cols B (fun j => NativeGraph.atom (xs j)) i) (NativeGraph.owner 1 B i.val)) : Cell NativeGraph.Word)=_
    have lowered := NativeGraph.source_graph_lowering inst 1 x.cols B
      (fun j => if hj : j<B then xs ⟨j,hj⟩ else 0) i.val i.isLt
    simpa [Floating.formula,SoftmaxTriton1.Floating.formula,reference,i.isLt] using congrArg (Cell.mk .real) lowered
  · intro r off miss
    rw [←hm]
    exact SoftmaxTriton1.Floating.row_frame (NativeGraph.algebra inst 1)
      y.region x.region x.offset y.offset x.rowStride x.colStride y.rowStride y.colStride
      x.cols B positive (Floating.launchState x.descriptor p s) u hu r off miss

end
end VeriTile.Bench.Repairs.TritonSoftmax.NativeIO
