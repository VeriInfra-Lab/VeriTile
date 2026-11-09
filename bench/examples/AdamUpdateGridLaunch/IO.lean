import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core
import VeriTile.Triton.DSL
import VeriTile.Triton.Float
import VeriTile.Triton.Launch.Composition
import VeriTile.Triton.Math.Optimizer
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Semantics
import bench.examples.AdamUpdateGridLaunch.Kernels
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch

open VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Examples

open VeriTile.Triton

set_option maxHeartbeats 5000000

open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- Masked in-place IO: the parameter and momentum buffers are both read and updated. -/
def adamIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := update_fn_kernel ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩ lr wd beta1 beta2 n B
  bufs := [⟨"p"⟩, ⟨"grad"⟩, ⟨"exp_avg"⟩]  -- p and exp_avg are updated in place
  in1 := ⟨"p"⟩
  in2 := ⟨"grad"⟩
  in3 := ⟨"exp_avg"⟩
  out1 := ⟨"p"⟩          -- = in1: in-place parameter update
  out2 := ⟨"exp_avg"⟩    -- = in3: in-place momentum update
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write1 := fun pid => pid * B
  write2 := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < n

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  { adamIO lr wd beta1 beta2 n B with kernel := optimizedKernel lr wd beta1 beta2 n B, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.AdamUpdateGridLaunch
