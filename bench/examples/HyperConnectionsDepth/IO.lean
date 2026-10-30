import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Memory.KernelSpec

/-! One shared complete scalar/matrix IO description. Mathematical proof views
replace only the interpreted kernel through these same layouts. -/
namespace VeriTile.Bench.Examples.HyperConnectionsDepth.IO
open VeriTile Triton
abbrev scalarLayout (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel)  : KernelIO₃ where
  kernel := kernel
  projection := projection
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


abbrev matrixLayout (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel) (S T D : Nat) : KernelIO₃ where
  kernel := kernel
  projection := projection
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


end VeriTile.Bench.Examples.HyperConnectionsDepth.IO
