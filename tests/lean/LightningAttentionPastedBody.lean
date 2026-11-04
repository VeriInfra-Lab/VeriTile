import bench.tritonbench_g.lightning_attention.Kernels

namespace VeriTile.Tests.LightningAttentionPastedBody
open VeriTile.Triton
open VeriTile.Bench.Repairs.LightningAttention
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly

/-- Preserve the original ordered ports, including the repeated inter reads.
The source collector's duplicates are meaningful metadata and stay intact. -/
example (p : Parameters) : (forward p).inputs=["query","key","value"] ∧
    (forward p).outputs=["output"] := ⟨rfl,rfl⟩
example (p : Parameters) : (intra p).inputs=["key","value","incoming","query"] ∧
    (intra p).outputs=["queryGradient","keyGradient","valueGradient"] := ⟨rfl,rfl⟩
example (p : Parameters) : (inter p).inputs=["incoming","queryGradient","value","key","key","value",
    "keyGradient","valueGradient","incoming","query"] ∧
    (inter p).outputs=["queryGradient","keyGradient","valueGradient"] := ⟨rfl,rfl⟩

end VeriTile.Tests.LightningAttentionPastedBody
