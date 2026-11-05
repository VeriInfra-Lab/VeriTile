import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit
import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import bench.examples.VectorAdd.Kernels
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.VectorAdd

open VeriTile.Bench.Examples.VectorAdd.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Triton

open VeriTile.Triton.KernelIO₂ (Implements)

open scoped VeriTile.Triton.KernelIO₂

open VeriTile.Examples

/-- Input/output windows for one aligned tile; allocation bounds remain quantified by the specification. -/
def addIO (B : Nat) : KernelIO₂ where
  kernel := addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (B : Nat) : KernelIO₂ :=
  { addIO B with kernel := optimizedKernel B, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.VectorAdd
