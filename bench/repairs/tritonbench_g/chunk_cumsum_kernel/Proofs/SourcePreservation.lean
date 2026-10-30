import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.SourcePreservation
open VeriTile.Triton
/-- Complete source identity for every earlier generic dtype model. No cast,
loop iteration, register, mask, undefined value or failure outcome is removed. -/
theorem original (dtype outType : FloatDType) (input output : RegionName) (T BT : Nat) :
    repairedKernel dtype outType input output T BT =
      LegacyKernels.repairedKernel dtype outType input output T BT := by
  cases dtype <;> cases outType <;> rfl

theorem candidate (dtype outType : FloatDType) (input output : RegionName) (T BT : Nat) :
    sumLeftKernel dtype outType input output T BT =
      LegacyKernels.sumLeftKernel dtype outType input output T BT := by
  cases dtype <;> cases outType <;> rfl

theorem actual_original (input output : RegionName) (T BT : Nat) :
    repairedKernel .fp32 .fp32 input output T BT = body input output T BT := rfl

theorem actual_candidate (input output : RegionName) (T BT : Nat) :
    sumLeftKernel .fp32 .fp32 input output T BT = sumLeftBody input output T BT := rfl
#axiomsClean original
#axiomsClean candidate
#axiomsClean actual_original
#axiomsClean actual_candidate
end VeriTile.Bench.Repairs.ChunkCumsumKernel.SourcePreservation
