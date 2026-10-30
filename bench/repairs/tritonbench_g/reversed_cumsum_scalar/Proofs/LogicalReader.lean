import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.IOComplete

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.IOComplete
open VeriTile.Triton Interface
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Full previous judgment iff the suffix formula on the selected logical row. -/
theorem logical_iff (h : Call) (c : Config) (row : Fin (h.B*h.H)) :
    (logicalIO h c row ⊨ fun xs t =>
      (∑ k : Fin h.T, if t.val ≤ k.val then xs k else 0 : ℝ)) ↔
    (io h c row).Implements (reference h) := Iff.rfl

end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.IOComplete
