import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fin.Rev
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core
import VeriTile.Triton.DSL
import VeriTile.Triton.Float
import VeriTile.Triton.Math.Reduction
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Semantics
import bench.examples.RowWiseSum.Kernels
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.ReductionSchedule
import VeriTile.Triton.Float.StructuralIO

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.RowWiseSum

open VeriTile.Bench.Examples.RowWiseSum.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?

open VeriTile.Triton

open VeriTile.Triton.KernelIO₁ (Implements)

open scoped VeriTile.Triton.KernelIO₁

open VeriTile.Examples

/-- Read B elements from row offset pid * nCol and write one sum at pid. -/
def rowWiseSumIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseSumKernel ⟨"x"⟩ ⟨"y"⟩ nCol B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := fun pid => pid

section Optimized

set_option maxHeartbeats 1600000

def reversedIO (nCol B : Nat) : KernelIO₁ :=
  { rowWiseSumIO nCol B with kernel := reversedKernel "x" "y" nCol B, projection := by rfl }

end Optimized

end VeriTile.Bench.Examples.RowWiseSum

namespace VeriTile.Bench.Examples.RowWiseSumFPEquiv

open VeriTile.Bench.Examples.RowWiseSum.Kernels

open VeriTile Triton

open FP.Structural FP.Equational

open FP.Guarded FP.ReductionSchedule FP.ScalarReduction

set_option maxHeartbeats 1600000

def originalIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseSumKernel "x" "y" nCol B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := id

def reversedIO (nCol B : Nat) : KernelIO₁ :=
  (originalIO nCol B).withKernel (reversedKernel "x" "y" nCol B)

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.RowWiseSumFPEquiv
