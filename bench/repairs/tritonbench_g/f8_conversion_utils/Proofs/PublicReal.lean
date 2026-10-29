import bench.repairs.tritonbench_g.f8_conversion_utils.IO
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.KernelContract
import VeriTile.Triton.Memory.KernelSpec.Masked

namespace VeriTile.Bench.Repairs.F8ConversionUtils.PublicReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.F8ConversionUtils
open Checked
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

/-- The IO assembler consumes the previously proved access-safety and
readback/frame results rather than proving a second source execution. -/
theorem implements (p : Source.Profile) (Y X : RegionName) (N B : Nat) :
    MaskedKernelIO₁.Implements
      (ConversionIO.real p Y X N B)
      (fun xs j => conversionReal (xs j)) := by
  unfold ConversionIO.real
  apply MaskedKernelIO₁.Implements.intro
  · unfold Kernel.FlattenOk
    rw [source_body]
    exact RealProof.flatten_ok p.direction.doubleStore Y X N B
  · intro bounds s input output _
    unfold Kernel.TraceSafe
    rw [source_body]
    exact RealProof.trace_safe p.direction.doubleStore Y X N B bounds s input output
  · intro s xs input
    obtain ⟨t,run,values,frame⟩ := RealSource.executes p.direction p.encoder p.format Y X N B s
    refine ⟨t,?_,?_,fun r i outside _ => frame r i outside⟩
    · simpa only [real_source] using run
    · intro j active
      exact (values j active).trans (input j active)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.F8ConversionUtils.PublicReal
