import bench.repairs.tritonbench_g.flash_decode2_phi.Kernels

/-! Canonical original and existing masked CSE bodies are shared with Llama in
flash_decode2_phi/Kernels.lean. Both retain every padded-lane mask. -/
namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked
abbrev kernel := FlashDecode2.phi
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Candidates
abbrev phiSharedBase := FlashDecode2.phiSharedBase
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Candidates
