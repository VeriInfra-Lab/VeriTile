import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.RealGrid
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.RealMath
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.RealSafety

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2500000

/-- Actual ordinary source-grid execution, source access safety from finite IO
bounds, independent causal attention, and the full original regional frame. -/
theorem complete (c : Call) (r : Regions) (shape : Host.Shape c) (labels : Labels r) :
    (io c r).ImplementsOn (domain c) (expected c) := by
  intro xs hd initial hin
  have hmeta := metadata c r xs hd initial hin
  obtain ⟨final,ran,outputs,frame⟩ := grid_result c r shape labels initial hmeta
  refine ⟨final,.cons ran (.nil final),?_,?_,?_⟩
  · intro bounds hb
    exact ⟨grid_safe c r xs initial hmeta bounds hb,trivial⟩
  · intro out j active
    fin_cases out
    exact (outputs (decode c j) active).trans (expected_matches c r xs initial hin (decode c j) active)
  · intro region off outside _
    exact (frame region off (outside_writes c r shape labels initial hmeta xs region off
      (fun j hj=>outside 0 j hj))).symm

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
