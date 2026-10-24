import bench.repairs.tritonbench_g.quant_transpose_kernel.IO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.Reference
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

/-- The two source-ordered FP32 multiplies, pinned signed64/FTZ conversion and
low8 store observation, for all native FP32 words including exceptional values. -/
def nativeTarget (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : Parameters) (version : Bool) :
    (IO.native multiply rest p version).Inputs → (IO.native multiply rest p version).Outputs :=
  IO.formula (nativeAlgebra multiply rest) nativeRound p version
    (fun x inverse => ((nativeByte multiply x inverse).toNat : Int))

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.Reference
