import bench.repairs.tritonbench_g.flash_decode2_phi.Kernels

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.Candidates
open VeriTile.Triton FP.Structural
noncomputable section

/-- Reuse the scalar split-row base for the vector base address, removing
one multiply and one add without changing the online floating recurrence. -/
def llamaSharedBase (H S D Q BD : Nat) : ComputeKernel :=
  let L : Region .nat := ⟨"L"⟩
  let M : RegionName := "M"
  let E : RegionName := "E"
  let O : RegionName := "O"
  triton {
  tl.region M = tl.float32, E = tl.float32, O = tl.float32
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BD))
  cur_batch_seq_len = tl.load(L + cur_batch)
  block_n_size = tl.where(cur_batch_seq_len <= $(0), $(0),
    cur_batch_seq_len + $(Q) - $(1)) // $(Q)
  sum_exp = 0.0
  max_logic = -inf
  acc = tl.zeros([$(BD)], dtype=tl.float32)
  offs_logic = cur_batch * $(H*S) + cur_head * $(S)
  offs_v = offs_logic * $(D) + offs_d
  for block_seq_n in range($(0), block_n_size, $(1)) {
    tv = tl.load(M + offs_v + block_seq_n * $(D))
    tlogic = tl.load(E + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(O + cur_batch * $(H*D) + cur_head * $(D) + offs_d,
    acc / sum_exp)
}

end
end VeriTile.Bench.Repairs.FlashDecode2Llama.Candidates
