import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.IOValues
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Mathematical
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open scoped BigOperators
noncomputable section

/-- Dense GEMV starts from the supplied output. The sentinel preserves it. -/
def real (h : Call) (indices : Nat → Int) (vals : ChannelValues ℝ (IO.inputs h indices)) :
    ChannelValues ℝ (IO.outputs h) :=
  Fin.cases (fun j => ofFloat h.outputType (
    let old := toFloat h.outputType (vals 3 j)
    if indices j.1.val = -1 then old else
      old+h.scaling*∑ k : Fin h.K,
        toFloat (dtype h.storage) (vals 1 (j.1,k,PUnit.unit))*
        toFloat (dtype h.storage) (vals 2
          (⟨(indices j.1.val).toNat,IO.selected_inside h indices j.1⟩,j.2.1,k,PUnit.unit))))
    (fun i => Fin.elim0 i)

/-- The established independent native recurrence preserves the actual half
multiply/reduce, FP32 accumulation and scaling, output conversion, and each
address's exact atomic event order. Arbitrary native input terms are admitted. -/
def native (plans : ScalarTree.Plans) (S : NativeGraph.Selection) (I : NativeGraph.Instructions)
    (h : Call) (indices : Nat → Int) (scale : NativeGraph.NG.Term) (orders : Orders h)
    (vals : ChannelValues NativeGraph.NG.Term (IO.inputs h indices)) : ChannelValues (BitVec 32) (IO.outputs h) :=
  Fin.cases (fun j => ofFloat h.outputType (NativeGraph.eval I S
    (Target.target NativeGraph.algebra (NativeGraph.primitives plans) h orders
      (IOValues.data h indices scale vals) j.1 j.2.1))) (fun i => Fin.elim0 i)

#specNonCircular real avoiding [Surface.source,FP.Structural.exec]
#specNonCircular native avoiding [Surface.source,FP.Structural.exec,FPValues.target]
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Mathematical
