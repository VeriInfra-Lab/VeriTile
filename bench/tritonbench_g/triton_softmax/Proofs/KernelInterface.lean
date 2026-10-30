import bench.tritonbench_g.triton_softmax.Proofs.KernelReal
import bench.tritonbench_g.triton_softmax.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.TritonSoftmax.RawKernel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1

/-- Raw region/stride geometry; no object identity, device, flags or Tensor laws. -/
structure View where
  region : RegionName
  rows : Nat
  cols : Nat
  offset : Nat
  rowStride : Nat
  colStride : Nat

def View.index (x : View) (r c : Nat) : Nat := x.offset+r*x.rowStride+c*x.colStride

def View.Injective (x : View) : Prop :=
  ∀ r c r' c',r<x.rows → c<x.cols → r'<x.rows → c'<x.cols →
    x.index r c=x.index r' c' → r=r' ∧ c=c'

/-- Internal adapter to the existing raw execution proof. All API-only fields
are fixed inert constants; callers provide only the six kernel geometry fields. -/
def View.descriptor (x : View) : Tensor :=
  ⟨0,x.region,x.rows,x.cols,x.offset,x.rowStride,x.colStride,0,Bool.false⟩

end VeriTile.Bench.Repairs.TritonSoftmax.RawKernel
