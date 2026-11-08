import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core
import VeriTile.Triton.DSL
import VeriTile.Triton.Float
import VeriTile.Triton.Math.Reduction
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Semantics
import bench.examples.RowWiseMax.Kernels
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.StructuralIO

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.RowWiseMax

open VeriTile.Bench.Examples.RowWiseMax.Kernels

open VeriTile.Triton

open VeriTile.Triton.KernelIO₁ (Implements)

open scoped VeriTile.Triton.KernelIO₁

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Examples

/-- Read B elements from row offset pid * nCol and write one maximum at pid. -/
def rowWiseMaxIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseMaxKernel ⟨"x"⟩ ⟨"y"⟩ nCol B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := fun pid => pid

section Optimized

set_option maxHeartbeats 1600000

def inlinedIO (nCol B : Nat) : KernelIO₁ :=
  { rowWiseMaxIO nCol B with kernel := inlinedKernel "x" "y" nCol B, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.RowWiseMax

namespace VeriTile.Bench.Examples.RowWiseMaxFPEquiv

open VeriTile.Bench.Examples.RowWiseMax.Kernels

open VeriTile Triton

open FP.Structural

set_option maxHeartbeats 1200000

def originalIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseMaxKernel "x" "y" nCol B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := id

def inlinedIO (nCol B : Nat) : KernelIO₁ :=
  (originalIO nCol B).withKernel (inlinedKernel "x" "y" nCol B)

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.RowWiseMaxFPEquiv
