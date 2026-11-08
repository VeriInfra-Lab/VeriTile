import VeriTile.Examples.ErasedProjection
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Meta.StatementAudit
import VeriTile.Triton
import bench.examples.FusedSwiglu.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.StructuralIO

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.FusedSwigluCorrect

open VeriTile.Bench.Examples.FusedSwiglu.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.MaskedKernelIO₂

def fusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := (swiglu_fused_kernel "x" "y" "out" n B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n

def unfusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := (swiglu_unfused "x" "y" "s" "out" n B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n
  scratch := [("s", fun pid => pid * B)]

end VeriTile.Bench.Examples.FusedSwigluCorrect

namespace VeriTile.Bench.Examples.FusedSwigluFPEquiv

open VeriTile.Bench.Examples.FusedSwiglu.Kernels

open VeriTile Triton

open FP.Structural

set_option maxHeartbeats 1200000

def unfusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := swiglu_unfused "X" "Y" "S" "OUT" n B
  in1 := "X"
  in2 := "Y"
  out := "OUT"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n
  scratch := [("S", fun pid => pid * B)]

def fusedIO (n B : Nat) : MaskedKernelIO₂ :=
  { (unfusedIO n B).withKernel (swiglu_fused_kernel "X" "Y" "OUT" n B) with scratch := [] }

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.FusedSwigluFPEquiv
