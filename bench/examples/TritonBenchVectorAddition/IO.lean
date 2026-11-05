import VeriTile.Meta.StatementAudit
import bench.examples.TritonBenchVectorAddition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_Correct
import VeriTile.Triton.Float.RewriteTactics

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect

open VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Independent IO wiring, program window and active-lane predicate. -/
def addIO (nElements blockSize : Nat) : MaskedKernelIO₂ where
  kernel := originalKernel nElements blockSize
  in1 := "x"
  in2 := "y"
  out := "output"
  B := blockSize
  read1 := fun pid => pid * blockSize
  read2 := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < nElements

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (n B : Nat) : MaskedKernelIO₂ :=
  { addIO n B with kernel := optimizedKernel n B, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
