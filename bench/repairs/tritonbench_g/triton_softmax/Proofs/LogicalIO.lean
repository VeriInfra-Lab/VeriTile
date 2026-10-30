import bench.repairs.tritonbench_g.triton_softmax.IO
import bench.repairs.tritonbench_g.triton_softmax.Proofs.LogicalColumns

namespace VeriTile.Bench.Repairs.TritonSoftmax.IO
open VeriTile.Triton
noncomputable section

theorem row_iff (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (cover : n≤B) :
    (rowIO out inp ib ob ir ic orow oc n B).Implements
      (fun (_,xs) i => Real.exp (xs.values i) / ∑ j : Fin n,Real.exp (xs.active cover j)) ↔
    (softmaxIO out inp ib ob ir ic orow oc n B).Implements
      (fun _ xs i => Real.exp (xs i) / ∑ j : Fin n,Real.exp (xs (Fin.castLE cover j,()))) := by rfl
end
end VeriTile.Bench.Repairs.TritonSoftmax.IO
