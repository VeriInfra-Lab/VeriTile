import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.adam_update_triton.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.AdamUpdateTriton
open VeriTile.Triton

/-! `source` is the canonical production body used by both public entries.
`engine` fixes every otherwise unannotated floating operation to FP32; physical
FP32 inputs and fixed native literal words are bound in Proofs/InputBinding
and Proofs/NativeWords. Runtime parameters are declared without executable
placeholder assignments. The full body is pasted literally from the selected Python source.
`mathematicalSource` remains solely an adapter for historical grid proofs. -/

inductive Output where
  | parameter | momentum
  deriving DecidableEq

structure Scalars (α : Type) where
  lr : α
  wd : α
  beta1 : α
  beta2 : α

/-- The total caller count remains the autotune key. Only the chunk length
is used in the original bounded per-program mask; scalar expression order is
the unchanged source's order. This specialization is the Real proof adapter. -/
def mathematicalSource (params : Scalars ℝ) (_total chunk block : Nat) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.AdamUpdateTriton.update_fn_kernel "P" "G" "M"
    params.lr params.wd params.beta1 params.beta2 chunk block

/-- External ABI for the exact selected Python body. Both element counts stay
runtime integers; the total count remains the host autotune key. Only the block
size is constexpr. All payload pointers and scalar argument registers are FP32.
The Nat call bindings use the existing mathematical index projection. -/
def sourceSignature : CallSignature where
  pointers := [("p_ptr",.fp32),("grad_ptr",.fp32),("exp_avg_ptr",.fp32)]
  scalars := [("lr",.fp32),("wd",.fp32),("beta1",.fp32),("beta2",.fp32),
    ("n_elements",.integer),("chunk_elements",.integer)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["p_ptr","grad_ptr","exp_avg_ptr"]
  writePorts := ["p_ptr","exp_avg_ptr"]

def sourceBody (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (n_elements chunk_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args
    [n_elements := n_elements, chunk_elements := chunk_elements] in
  triton {
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < chunk_elements

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

def commutedBody (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (n_elements chunk_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  with_triton_call sourceSignature call_args
    [n_elements := n_elements, chunk_elements := chunk_elements] in
  triton {
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < chunk_elements

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

/-- Compatibility invocation for callers without the unused autotune key.
The selected full invocation below supplies its actual total element count. -/
def sourceWithArgumentDeclarations (P G M : RegionName) (chunk block : Nat) : ComputeKernel :=
  sourceBody P G M 0 chunk block

def commutedWithArguments (P G M : RegionName) (chunk block : Nat) : ComputeKernel :=
  commutedBody P G M 0 chunk block

/-- The actual selected original and candidate retain every source statement,
including bool-to-FP32 multiplication and both in-place stores. -/
def source (total chunk block : Nat) : ComputeKernel :=
  sourceBody "P" "G" "M" total chunk block

def commutedSource (total chunk block : Nat) : ComputeKernel :=
  commutedBody "P" "G" "M" total chunk block

def engine {α : Type} (algebra : FP.Structural.Algebra α) : FP.Structural.Algebra α :=
  algebra.withDefaultPrecision .fp32

/-- Runtime scalar ABI values occupy the four scalar argument registers.
No Real decoding, finite-range premise or whole-output numerical law occurs. -/
def bindArguments {α : Type} [Inhabited α] (params : Scalars α)
    (state : FP.Structural.State α) : FP.Structural.State α :=
  let state := state.setReg "lr" .real [] (fun _ => params.lr)
  let state := state.setReg "wd" .real [] (fun _ => params.wd)
  let state := state.setReg "beta1" .real [] (fun _ => params.beta1)
  state.setReg "beta2" .real [] (fun _ => params.beta2)

end VeriTile.Bench.Repairs.AdamUpdateTriton
