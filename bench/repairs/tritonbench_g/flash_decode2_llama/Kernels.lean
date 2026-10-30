import bench.repairs.tritonbench_g.flash_decode2_phi.Kernels

/-! Canonical original and existing CSE bodies are shared with Phi in
flash_decode2_phi/Kernels.lean. Both use the unchanged wrapper stride ABI. -/
namespace VeriTile.Bench.Repairs.FlashDecode2Llama.Candidates
abbrev llamaSharedBase := FlashDecode2.llamaSharedBase
end VeriTile.Bench.Repairs.FlashDecode2Llama.Candidates
