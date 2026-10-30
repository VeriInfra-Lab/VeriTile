import VeriTile.Triton.DSL

/-! Complete masked in-place fp32 source from fused_activation.py and a candidate
commuting its final bias addition. Both sigmoid and ReLU branches are retained.
The original source has no explicit dtype overrides; this selected version
uses typed FP32 regions. The inner multiply/add contraction is retained in
the separate native graph interpretation, never assumed equal to split rounding.
The host launch is outside these per-program contracts. -/

namespace VeriTile.Bench.Repairs.FusedActivation
open VeriTile.Triton

/-- Faithful transcription of `fused_activation.py`'s
`fused_add_mul_activation_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `num_weights/xnumel/multiplier/activation/BLOCK_SIZE: tl.constexpr` ->
  Lean parameters.
- Python `activation == "sigmoid"` / `elif activation == "relu"` -> Lean
  Boolean `activation`; `true` selects sigmoid and `false` selects ReLU. -/
def kernel
    (X B I : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let bias_ptr : Region .fp32 := B.cast
  let in_ptr : Region .fp32 := I.cast
  triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))[:]
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
  tmp3 = tl.load(in_ptr + index, mask)
  activ_input = $(multiplier) * tmp3 + tmp0 + tmp1
  if activation {
    ma_result = tl.sigmoid(activ_input)
  } else {
    ma_result = tl.maximum(0, activ_input)
  }
  tl.store(x_ptr + index, ma_result, mask)
}

/-- Commute the combined input and the loaded bias, retaining all memory operations. -/
def commutedKernel
    (X B I : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let bias_ptr : Region .fp32 := B.cast
  let in_ptr : Region .fp32 := I.cast
  triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))[:]
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
  tmp3 = tl.load(in_ptr + index, mask)
  activ_input = tmp1 + ($(multiplier) * tmp3 + tmp0)
  if activation {
    ma_result = tl.sigmoid(activ_input)
  } else {
    ma_result = tl.maximum(0, activ_input)
  }
  tl.store(x_ptr + index, ma_result, mask)
}

end VeriTile.Bench.Repairs.FusedActivation
