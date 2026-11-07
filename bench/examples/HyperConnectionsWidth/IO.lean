import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core
import VeriTile.Triton.DSL
import VeriTile.Triton.Float
import VeriTile.Triton.Math.MatrixRewrite
import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Semantics
import bench.examples.HyperConnectionsWidth.Kernels
import VeriTile.Triton.Float.FP32ReferenceRewrite
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.HyperConnectionsWidth

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Triton

open VeriTile.Triton.KernelIO₃ₓ₂ (Implements)

open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-- Scalar specialization S = T = D = 1: a per-program residual, shared logits and two output cells. -/
@[reducible] def mhcWidthIO (tau : ℝ) : KernelIO₃ₓ₂ where
  kernel := mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
    ⟨"branch_in"⟩ 1 1 1 0 tau
  in1 := ⟨"res"⟩
  in2 := ⟨"h_res"⟩
  in3 := ⟨"h_pre"⟩
  out1 := ⟨"res_mix"⟩
  out2 := ⟨"branch_in"⟩
  B1 := 1
  B2 := 1
  B3 := 1
  Bout1 := 1
  Bout2 := 1
  read1 := fun pid => pid        -- each program reads its own residual cell
  read2 := fun _ => 0            -- ALL programs read the shared scalar logit
  read3 := fun _ => 0
  write1 := fun pid => pid
  write2 := fun pid => pid

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ₓ₂ :=
  { mhcWidthIO tau with kernel := optimizedKernel tau, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.HyperConnectionsWidth

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.MatrixCorrect

open VeriTile Triton LogSinkhorn

open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels

set_option maxHeartbeats 6400000

set_option linter.unusedSimpArgs false

open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-- Both output matrices have their own bounded row-major windows. -/
def matrixIO (optimized : Bool) (S T D iters : Nat) (tau : ℝ) : KernelIO₃ₓ₂ where
  kernel := if optimized then matrixOptimizedKernel "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau
    else matrixOriginal S T D iters tau
  projection := by cases optimized <;> rfl
  in1 := "res"
  in2 := "h_res"
  in3 := "h_pre"
  out1 := "res_mix"
  out2 := "branch_in"
  B1 := S * D
  B2 := S * S
  B3 := S * T
  Bout1 := S * D
  Bout2 := T * D
  read1 := fun pid => pid * (S * D)
  read2 := fun _ => 0
  read3 := fun _ => 0
  write1 := fun pid => pid * (S * D)
  write2 := fun pid => pid * (T * D)

end VeriTile.Bench.Examples.HyperConnectionsWidth.MatrixCorrect
