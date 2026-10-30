import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOImplements
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealFormula
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TargetBridge

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PublicIO
open VeriTile.Triton FP.Structural FPValues NativeGraph
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section

theorem real (h : Call) (indices : Nat → Int) (orders : Orders h) :
    (IO.programIO Real.algebra Real.primitives h indices h.scaling orders id).ImplementsOnCells
      (IO.domain h indices) (Mathematical.real h indices) := by
  apply IOImplements.implements
  exact RealFormula.real_formula h indices orders

theorem native (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (h : Call) (indices : Nat → Int)
    (scale : NG.Term) (orders : Orders h) :
    (IO.programIO algebra (primitives plans) h indices scale orders
      (fun term => (binding.program term).eval I)).ImplementsOnCells
        (IO.domain h indices) (Mathematical.native plans S I h indices scale orders) := by
  apply IOImplements.implements
  intro vals b n
  rw [compiled_expression,Target.agrees]
  rfl

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PublicIO
