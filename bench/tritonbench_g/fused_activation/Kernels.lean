import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.FusedActivation
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("bias_ptr",.fp32),("in_ptr",.fp32)]
  scalars := []
  constexpr := ["num_weights", "xnumel", "multiplier", "activation", "BLOCK_SIZE"]
  readPorts := ["x_ptr", "bias_ptr", "in_ptr"]
  writePorts := ["x_ptr"]

/-- Unchanged selected Python body `fused_add_mul_activation_kernel`. The existing Boolean API supplies
one of the two valid source String constexpr values before the first statement. -/
def kernel (X B I : RegionName) (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (useSigmoid : Bool) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let bias_ptr : Region .fp32 := B.cast
  let in_ptr : Region .fp32 := I.cast
  let activation := if useSigmoid then "sigmoid" else "relu"
  with_triton_signature callSignature in
  triton {
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    bias_index = index % num_weights
    tmp0 = tl.load(x_ptr + index, mask)
    tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr + index, mask)
    activ_input = multiplier * tmp3 + tmp0 + tmp1
    if activation == "sigmoid":
        ma_result = tl.sigmoid(activ_input)
        # option 2 - calculate sigmoid using exp
        # ma_result = 1.0 / (1.0 + tl.exp(-sigmoid_input))
        # option 3: fast sigmoid - inaccurate but faster
        # ma_result = 1.0 / (1.0 + tl.abs(sigmoid_input))
    elif activation == "relu":
        ma_result = tl.maximum(0, activ_input)

    tl.store(x_ptr + index, ma_result, mask)
  }

/-- Unchanged selected Python body `fused_add_mul_activation_commuted_kernel`. The existing Boolean API supplies
one of the two valid source String constexpr values before the first statement. -/
def commutedKernel (X B I : RegionName) (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (useSigmoid : Bool) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let bias_ptr : Region .fp32 := B.cast
  let in_ptr : Region .fp32 := I.cast
  let activation := if useSigmoid then "sigmoid" else "relu"
  with_triton_signature callSignature in
  triton {
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    bias_index = index % num_weights
    tmp0 = tl.load(x_ptr + index, mask)
    tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr + index, mask)
    activ_input = tmp1 + (multiplier * tmp3 + tmp0)
    if activation == "sigmoid":
        ma_result = tl.sigmoid(activ_input)
        # option 2 - calculate sigmoid using exp
        # ma_result = 1.0 / (1.0 + tl.exp(-sigmoid_input))
        # option 3: fast sigmoid - inaccurate but faster
        # ma_result = 1.0 / (1.0 + tl.abs(sigmoid_input))
    elif activation == "relu":
        ma_result = tl.maximum(0, activ_input)

    tl.store(x_ptr + index, ma_result, mask)
  }

end VeriTile.Bench.Repairs.FusedActivation
