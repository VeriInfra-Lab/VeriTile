import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- The integer domain is transported separately from checked INT32 lengths and
INT64 promoted native addresses. Floating storage and arithmetic remain FP32. -/
def phi (H S D Q BD : Nat) : ComputeKernel :=
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
  offs_v = cur_batch * $(H*S*D) + cur_head * $(S*D) + offs_d
  offs_logic = cur_batch * $(H*S) + cur_head * $(S)
  for block_seq_n in range($(0), block_n_size, $(1)) {
    tv = tl.load(M + offs_v + block_seq_n * $(D), mask=offs_d < $(D), other=0.0)
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
    acc / sum_exp, mask=offs_d < $(D))
}

def llama (H S D Q BD : Nat) : ComputeKernel :=
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
  offs_v = cur_batch * $(H*S*D) + cur_head * $(S*D) + offs_d
  offs_logic = cur_batch * $(H*S) + cur_head * $(S)
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

def kernel (masked : Bool) (H S D Q BD : Nat) : ComputeKernel :=
  if masked then phi H S D Q BD else llama H S D Q BD

def prologue (H S D Q BD : Nat) : List ComputeStmt :=
  (phi H S D Q BD).surfaceBody.take 10

def body (masked : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  match (kernel masked H S D Q BD).surfaceBody.getD 10 (.effectMarker "absent") with
  | ComputeStmt.forRangeDyn _ _ _ _ b => b
  | _ => []

def epilogue (masked : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  (kernel masked H S D Q BD).surfaceBody.drop 11


end
end VeriTile.Bench.Repairs.FlashDecode2
