import bench.examples.HyperConnectionsWidth.Kernels
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Memory.KernelSpec

/-! One shared complete scalar/matrix IO description. Mathematical proof views
replace only the interpreted kernel through these same layouts. -/
namespace VeriTile.Bench.Examples.HyperConnectionsWidth.IO
open VeriTile Triton
abbrev scalarLayout (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel)  : KernelIO₃ₓ₂ where
  kernel := kernel
  projection := projection
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


abbrev matrixLayout (kernel : ComputeKernel)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel) (S T D : Nat) : KernelIO₃ₓ₂ where
  kernel := kernel
  projection := projection
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


end VeriTile.Bench.Examples.HyperConnectionsWidth.IO
