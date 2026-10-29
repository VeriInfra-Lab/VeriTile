import bench.examples.AdamUpdateGridLaunch.Kernels
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Launch.Grid

/-! One complete masked in-place IO layout, shared by the source and its
mathematical proof interpretation. Both parameter and momentum stores remain. -/
namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch.IO
open VeriTile Triton

abbrev adamGrid (m : Nat) : Grid := { dims := [m] }
abbrev layout (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel) (n B : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := kernel
  projection := projection
  bufs := [⟨"p"⟩, ⟨"grad"⟩, ⟨"exp_avg"⟩]  -- p and exp_avg are updated in place
  in1 := ⟨"p"⟩
  in2 := ⟨"grad"⟩
  in3 := ⟨"exp_avg"⟩
  out1 := ⟨"p"⟩          -- = in1: in-place parameter update
  out2 := ⟨"exp_avg"⟩    -- = in3: in-place momentum update
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write1 := fun pid => pid * B
  write2 := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < n

end VeriTile.Bench.Examples.AdamUpdateGridLaunch.IO
