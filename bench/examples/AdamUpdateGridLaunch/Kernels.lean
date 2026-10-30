import VeriTile.Triton.DSL.CallSignature

/-! Original source: bench/tritonbench_g/adam_update_triton/adam_update_triton.py,
update_fn_kernel. Candidate commutes only the final momentum addition.
FP32 pointers/scalars and caller-bound runtime element count stay outside both
unchanged pasted bodies; BLOCK_SIZE alone is constexpr. Nat call values use the
existing mathematical index projection, not a new native overflow guarantee.
Separate arithmetic uses enable_fp_fusion=False. -/
namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels
open VeriTile Triton

def updateSignature : CallSignature where
  pointers := [("p_ptr",.fp32),("grad_ptr",.fp32),("exp_avg_ptr",.fp32)]
  scalars := [("lr",.fp32),("wd",.fp32),("beta1",.fp32),("beta2",.fp32),("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["p_ptr","grad_ptr","exp_avg_ptr"]
  writePorts := ["p_ptr","exp_avg_ptr"]

def update_fn_kernel (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call updateSignature call_args [lr := lr, wd := wd, beta1 := beta1, beta2 := beta2, n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Stepweight decay
    p = p * (1 - lr * wd)

    # Diff between momentum running average and grad
    diff = exp_avg - grad

    # Weight update
    update = diff * beta1 + grad

    # torch.sign
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # Decay the momentum running average coefficient
    exp_avg = diff * beta2 + grad

    # Store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
  }

def commuted_update (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call updateSignature call_args [lr := lr, wd := wd, beta1 := beta1, beta2 := beta2, n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Stepweight decay
    p = p * (1 - lr * wd)

    # Diff between momentum running average and grad
    diff = exp_avg - grad

    # Weight update
    update = diff * beta1 + grad

    # torch.sign
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # Decay the momentum running average coefficient
    exp_avg = grad + diff * beta2

    # Store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
  }

def originalKernel (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  update_fn_kernel "p" "grad" "exp_avg" lr wd beta1 beta2 n_elements BLOCK_SIZE

def optimizedKernel (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  commuted_update "p" "grad" "exp_avg" lr wd beta1 beta2 n_elements BLOCK_SIZE
end VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels
