import bench.examples.FusedSwiglu.Kernels
import VeriTile.Triton.Float.Kernel

/-! Shared full masked IO: all n/B including empty and partial tiles; candidate
scratch is private BF16 storage. The Real view uses lowercase region names. -/
namespace VeriTile.Bench.Examples.FusedSwiglu.IO
open VeriTile Triton
open VeriTile.Bench.Examples.FusedSwiglu.Kernels

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
  { (unfusedIO n B).withKernel (swiglu_fused "X" "Y" "OUT" n B) with scratch := [] }


end VeriTile.Bench.Examples.FusedSwiglu.IO
