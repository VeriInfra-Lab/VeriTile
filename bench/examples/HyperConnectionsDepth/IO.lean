import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core
import VeriTile.Triton.DSL
import VeriTile.Triton.Float
import VeriTile.Triton.Math.MatrixRewrite
import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Semantics
import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.HyperConnectionsDepth

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Triton

open VeriTile.Triton.KernelIO₃ (Implements)

open scoped VeriTile.Triton.KernelIO₃

/-- Scalar specialization S = T = D = 1: per-program residual/branch cells and a shared logit. -/
@[reducible] def mhcDepthIO (tau : ℝ) : KernelIO₃ where
  kernel := mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
    ⟨"out"⟩ 1 1 1 0 tau
  in1 := ⟨"res_mix"⟩
  in2 := ⟨"branch_out"⟩
  in3 := ⟨"h_post"⟩
  out := ⟨"out"⟩
  B1 := 1
  B2 := 1
  B3 := 1
  Bout := 1
  read1 := fun pid => pid        -- each program reads its own residual-mix cell
  read2 := fun pid => pid        -- … and its own branch-output cell
  read3 := fun _ => 0            -- ALL programs read the shared scalar logit
  write := fun pid => pid

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ :=
  { mhcDepthIO tau with kernel := optimizedKernel tau, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.HyperConnectionsDepth

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect

open VeriTile Triton LogSinkhorn

open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

set_option maxHeartbeats 3200000

open scoped VeriTile.Triton.KernelIO₃

/-- Row-major input/output windows for either complete matrix source. -/
def matrixIO (optimized : Bool) (S T D iters : Nat) (tau : ℝ) : KernelIO₃ where
  kernel := if optimized then matrixOptimizedKernel "res_mix" "branch_out" "h_post" "out" S T D iters tau
    else matrixOriginal S T D iters tau
  projection := by cases optimized <;> rfl
  in1 := "res_mix"
  in2 := "branch_out"
  in3 := "h_post"
  out := "out"
  B1 := S * D
  B2 := T * D
  B3 := T * S
  Bout := S * D
  read1 := fun pid => pid * (S * D)
  read2 := fun pid => pid * (T * D)
  read3 := fun _ => 0
  write := fun pid => pid * (S * D)

end VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect
