import bench.repairs.tritonbench_g.triton_softmax.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.TritonSoftmax.KernelResult
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
open RawKernel
noncomputable section

/-- Complete shared-source execution, independent unshifted softmax and raw cell frame. -/
theorem kernel_real (x y : View) (B p : Nat)
    (shape : y.rows=x.rows ∧ y.cols=x.cols) (injective : y.Injective)
    (positive : 0<B) (cover : x.cols≤B) (row : p<x.rows)
    (s : BlockState) (values : Nat → Nat → ℝ)
    (input : ∀ r c,r<x.rows → c<x.cols → s.readMem x.region (x.index r c)=values r c) :
    ∃ t,RealModel.evaluate x.descriptor y.descriptor B p s=some t ∧
      (∀ c,c<x.cols → t.readMem y.region (y.index p c)=
        Real.exp (values p c)/(∑ j∈Finset.range x.cols,Real.exp (values p j))) ∧
      (∀ r i,(∀ c,c<y.cols → ¬(y.region=r ∧ y.index p c=i)) → t.mem r i=s.mem r i) := by
  obtain ⟨t,run⟩ := RealModel.terminates x.descriptor y.descriptor B p positive s
  have result := RealModel.source.row x.descriptor y.descriptor B p s t values
    ⟨shape.1,shape.2,rfl⟩ injective positive cover row input run
  refine ⟨t,run,result.1,?_⟩
  intro r i outside
  exact result.2 r i (by
    intro q hq c hc
    have same : q=p := by simpa using hq
    subst q
    exact outside c hc)

#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.TritonSoftmax.KernelResult
