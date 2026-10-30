import bench.repairs.tritonbench_g.reversed_cumsum.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ReversedCumsum.Kernels
theorem configurations_positive (c : Config) (hc : c∈configurations) : 0<c.BT ∧ c.BT≤64 := by
  simp [configurations] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem source_projects (h : Call) (c : Config) :
    (source h c).toAlgorithm?=.ok (source h c).toAlgKernel := by
  cases h.inputDType <;> cases h.outputDType <;> rfl

#axiomsClean source_projects
end VeriTile.Bench.Repairs.ReversedCumsum.Kernels
