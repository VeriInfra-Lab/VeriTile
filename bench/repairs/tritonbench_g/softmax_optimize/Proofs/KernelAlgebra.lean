import bench.repairs.tritonbench_g.softmax_optimize.Proofs.FPFrame
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.NativeOrder

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.KernelNative
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native (Term Word Instructions Schedule Reductions)
noncomputable section

/-- IEEE equality with the source's -infinity sentinel is exact word equality.
The callback is used only by the two actual source guards. -/
def equal (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder) (a b : Term) : Bool :=
  decide (NativeOrder.word I S O a=NativeOrder.word I S O b)

def algebra (R : Reductions) (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder) : Algebra Term :=
  FPExecution.withEq (VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.algebra R) (equal I S O)

end
end VeriTile.Bench.Repairs.SoftmaxOptimize.KernelNative
