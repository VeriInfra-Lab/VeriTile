import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Safety

/-! Public kernel-only admission is `KernelOnly.Contract`: actual shared FP32
Tensor API bindings, the selected source's unchanged dimensions/pointer guards,
and fresh disjoint output backings. `Preparation` states only the ordinary
incoming contiguous-copy input binding and admits noncontiguous/broadcast G.
No host lifecycle, allocator availability or successful execution is assumed. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
#specNonCircular Contract avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run]
#specNonCircular Preparation avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
