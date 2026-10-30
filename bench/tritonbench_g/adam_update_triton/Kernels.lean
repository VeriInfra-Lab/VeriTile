import bench.tritonbench_g.adam_update_triton.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.AdamUpdateTriton
open VeriTile.Triton

/-! `source` is the canonical production body used by both public entries.
`engine` fixes every otherwise unannotated floating operation to FP32; physical
FP32 inputs and fixed native literal words are bound in Proofs/InputBinding
and Proofs/NativeWords. Runtime parameters are declared without executable
placeholder assignments. There is no handwritten body AST.
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

/-- Runtime FP32 scalar arguments use the existing real-valued carrier.
The declarations emit no executable statements; the ABI supplies registers. -/
def sourceWithArgumentDeclarations (P G M : RegionName) (chunk block : Nat) : ComputeKernel := triton {
  tl.param lr : tl.float32
  tl.param wd : tl.float32
  tl.param beta1 : tl.float32
  tl.param beta2 : tl.float32
  pid = tl.program_id(axis=0)
  block_start = pid * $(block)
  offsets = block_start + tl.arange(0, $(block))
  mask = offsets < $(chunk)
  offset_p_ptr = P + offsets
  offset_grad_ptr = G + offsets
  offset_exp_avg_ptr = M + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - lr * wd)
  diff = exp_avg - grad
  update = diff * beta1 + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -lr, lr)
  p = p + update_sign * tl.where(can_update, $((1 : ℝ)), $((0 : ℝ)))
  exp_avg = diff * beta2 + grad
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}

/-- Existing momentum-addition commutation with the same runtime ABI. -/
def commutedWithArguments (P G M : RegionName) (chunk block : Nat) : ComputeKernel := triton {
  tl.param lr : tl.float32
  tl.param wd : tl.float32
  tl.param beta1 : tl.float32
  tl.param beta2 : tl.float32
  pid = tl.program_id(axis=0)
  block_start = pid * $(block)
  offsets = block_start + tl.arange(0, $(block))
  mask = offsets < $(chunk)
  offset_p_ptr = P + offsets
  offset_grad_ptr = G + offsets
  offset_exp_avg_ptr = M + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - lr * wd)
  diff = exp_avg - grad
  update = diff * beta1 + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -lr, lr)
  p = p + update_sign * tl.where(can_update, $((1 : ℝ)), $((0 : ℝ)))
  exp_avg = grad + diff * beta2
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}


/-- The actual source body with runtime (not Real-literal-encoded) scalars.
All three FP32 payloads use the source-selected default FP32 engine.
The bool multiplicand is explicitly promoted by where(1,0), preserving the
source FP multiplication instead of the DSL's Real-only boolean-mask rewrite. -/
def source (_total chunk block : Nat) : ComputeKernel :=
  sourceWithArgumentDeclarations "P" "G" "M" chunk block

/-- Same P/G/M inputs and in-place P/M outputs; only final addition order changes. -/
def commutedSource (_total chunk block : Nat) : ComputeKernel :=
  commutedWithArguments "P" "G" "M" chunk block

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
