import VeriTile.Examples.ErasedProjection
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Meta.StatementAudit
import VeriTile.Triton
import bench.examples.FusedSiLU.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.StructuralIO

/-! The logical and floating-point program interfaces. -/

namespace VeriTile.Bench.Examples.FusedSiLUCorrect

open VeriTile.Bench.Examples.FusedSiLU.Kernels

open VeriTile Triton

open scoped VeriTile.Triton.KernelIO₃

def fusedIO (B : Nat) : KernelIO₃ where
  kernel := (fusedSiLUKernel "x" "gate" "residual" "out" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B

def unfusedIO (B : Nat) : KernelIO₃ where
  kernel := (unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B).eraseDType
  projection := ComputeKernel.eraseDType_projectable _ (by rfl)
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [⟨"z", fun pid => pid * B, B⟩, ⟨"silu", fun pid => pid * B, B⟩]

end VeriTile.Bench.Examples.FusedSiLUCorrect

namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv

open VeriTile.Bench.Examples.FusedSiLU.Kernels

open VeriTile Triton

open FP.Structural

set_option maxHeartbeats 1200000

def unfusedIO (B : Nat) : KernelIO₃ where
  kernel := unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [{ buf := "z", win := fun pid => pid * B, len := B },
    { buf := "silu", win := fun pid => pid * B, len := B }]

def fusedIO (B : Nat) : KernelIO₃ :=
  { (unfusedIO B).withKernel (fusedSiLUKernel "x" "gate" "residual" "out" B) with
    scratch := [] }

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.FusedSiLUFPEquiv
