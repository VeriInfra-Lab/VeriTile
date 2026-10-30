import VeriTile.Triton.Float.PerAddressProgramEquivalenceSyntax
import bench.tritonbench_g.bgmv_shrink_kernel.IO

/-! Pure candidate invocation on the shared per-address IO. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PairIO
open VeriTile Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section
variable {A : Type} [Inhabited A]

/-- Change only the complete kernel. Runtime scale, launch dimensions, actual
program IDs, backend and cleared register file are retained verbatim. -/
def invocation (M : Algebra A) (P : Primitives A) (h : Call) (scale : A)
    (p : Program h) : Invocation A :=
  { IO.invocation M P h scale p with kernel := Kernels.scalarLeft h }

def candidate (M : Algebra A) (P : Primitives A) (h : Call)
    (indices : Nat → Int) (scale : A) (orders : Orders h) :
    PerAddressProgramIO.Variant (IO.programIO M P h indices scale orders id) where
  invocation := fun _ => invocation M P h scale
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PairIO
