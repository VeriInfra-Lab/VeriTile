import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama
open VeriTile Triton FP.Structural
/-- Compatibility proof decomposition. Actual candidate programs are complete
ordinary quotations in Kernels.lean. -/
def indexCommuted (source : ComputeKernel) (Y : RegionName) (N : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs
    (source.surfaceBody.take 1 ++
      [.assign .ptr [] "Y" (.alg (.ptrAdd .nil (.ptrBase Y)
        (.mul .nat .nil (.constNat N) (.ref .nat [] "row"))))] ++ source.surfaceBody.drop 2)

end VeriTile.Bench.Repairs.RmsnormFusedLlama
