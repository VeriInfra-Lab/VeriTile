import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit
import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import bench.examples.FlatVectorAdd.Kernels
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.FlatVectorAdd

open VeriTile.Bench.Examples.FlatVectorAdd.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Triton

open VeriTile.Triton.MaskedKernelIO₂ (Implements)

open scoped VeriTile.Triton.MaskedKernelIO₂

open VeriTile.Examples (observeAt)

/-- Masked tile windows: only lanes with pid * B + i < n are active. -/
def addMaskedIO (B n : Nat) : MaskedKernelIO₂ where
  kernel := addMaskedKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < n

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (B n : Nat) : MaskedKernelIO₂ :=
  { addMaskedIO B n with kernel := optimizedKernel n B, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.FlatVectorAdd
