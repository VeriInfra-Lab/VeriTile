import bench.tritonbench_g.square_matrix.Contract
import bench.tritonbench_g.flash_decode2_phi.Proofs.Contract

namespace VeriTile.Bench.Repairs.FlashDecode2Llama
abbrev Call := {c : FlashDecode2.Checked.Call // c.masked = Bool.false}
abbrev InputContract (c : Call) := FlashDecode2.Checked.InputContract c.val
abbrev checkInput (c : Call) := FlashDecode2.Checked.checkInput c.val
abbrev checkArguments := FlashDecode2.Checked.checkArguments Bool.false
end VeriTile.Bench.Repairs.FlashDecode2Llama
