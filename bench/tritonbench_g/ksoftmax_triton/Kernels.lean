import bench.tritonbench_g.ksoftmax_triton.Sources.Kernels

/-! Transparent reuse of the complete ordinary forward and backward quotations
in the shared source entry; only the selected Python host guard differs. -/
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked
/-- The guard-only candidate uses the frozen complete source quotations. -/
abbrev forward := Ksoftmax.forward
abbrev backward := Ksoftmax.backward

theorem forward_unchanged : forward=Ksoftmax.forward := rfl
theorem backward_unchanged : backward=Ksoftmax.backward := rfl
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked
