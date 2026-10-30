import bench.tritonbench_g.context_attn_mistral.Proofs.Checked

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel

/-- The retained checked kernel input domain, including underlaunch, grouped
heads, signed metadata, dense storage, pointer bounds and output separation. -/
abbrev Contract := VeriTile.Bench.Repairs.ContextAttnMistral.Checked.Admission
noncomputable abbrev checkInputs := VeriTile.Bench.Repairs.ContextAttnMistral.Checked.check
alias checkInputs_iff := VeriTile.Bench.Repairs.ContextAttnMistral.Checked.check_iff

#axiomsClean checkInputs_iff

end VeriTile.Bench.Repairs.ContextAttnMistralKernel
