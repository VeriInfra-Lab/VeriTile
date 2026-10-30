import bench.tritonbench_g.chunk_gated_attention.IO

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section

theorem capacity_bound {capacity extent : Nat}
    (h : ∀ j : Fin capacity, j.val < extent) : capacity ≤ extent := by
  by_contra hn
  exact Nat.lt_irrefl extent (h ⟨extent,Nat.lt_of_not_ge hn⟩)

theorem trace_safe {stages : List AlgorithmStage} {before after : BlockState}
    (trace : AlgorithmExecution stages before after) (bounds : RegionBounds)
    (safe : ∀ stage ∈ stages, ∀ s, stage.Safe bounds s) : trace.Safe bounds := by
  induction trace with
  | nil => trivial
  | cons head tail ih =>
    exact ⟨safe _ List.mem_cons_self _,ih (fun stage hm => safe stage (List.mem_cons_of_mem _ hm))⟩

theorem prefix_trace (p : RealCumProgram.Layout) (heads : Nat) (input output : RegionName)
    {plans : List LaunchPlan.Item} {before after : BlockState}
    (run : DirectCumulativeSequence.Runs p heads input output Bool.false .fp32 .fp32 plans before after) :
    Nonempty (AlgorithmExecution (prefixStages p heads input output plans) before after) := by
  induction run with
  | nil => exact ⟨.nil _⟩
  | cons plan rest u mid v head tail ih =>
    obtain ⟨trace⟩ := ih
    exact ⟨.cons (by simpa [AlgorithmStage.Runs,selected_projection] using head) trace⟩

theorem inner_trace (p : RealHMath.Layout) (r : RealHProgram.Regions)
    (gateK hasSeed storeFinal : Bool) {plans : List LaunchPlan.Item} {before after : BlockState}
    (run : DirectRealSequence.Runs p r .fp32 .fp32 .fp32 .fp32 .fp32
      gateK hasSeed storeFinal Bool.false plans before after) :
    Nonempty (AlgorithmExecution (innerStages p r gateK hasSeed storeFinal plans) before after) := by
  induction run with
  | nil => exact ⟨.nil _⟩
  | cons plan rest u mid v head tail ih =>
    obtain ⟨trace⟩ := ih
    exact ⟨.cons (by simpa [AlgorithmStage.Runs,selected_projection] using head) trace⟩

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
