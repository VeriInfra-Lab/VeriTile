import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeProgramBridge
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Safety
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Native

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 4000000

/-- All five source metadata roles are the checked words of the original heap. -/
theorem full_metadata (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    Integer.FullMetadata r.call (initial r heap) := by
  refine ⟨initial_metadata r heap h,?_,?_⟩
  · intro b hb
    simpa [initial,Request.call,regions] using (h.checks.2.2.2.1 ⟨b,hb⟩).2.2.2.1
  · intro b hb j hj
    simpa [initial,Request.call,regions] using (h.checks.2.2.2.1 ⟨b,hb⟩).2.2.2.2 ⟨j,hj⟩

theorem reference_frame (r : Request) (jobs : List (GridIndex r.call.grid))
    (s : State NativeGraph.Expr) (rg : RegionName) (off : Nat) (hne : rg≠regions.out) :
    (FPReference.grid r.call (backend r) (NativeInstructions.support r.call) jobs s).mem rg off=s.mem rg off := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    rw [FPReference.grid,ih]
    exact FPSuccess.scatter_frame _ _ _ _ _ rg off (by intro i on same _; exact hne same)

theorem prefix_metadata (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (jobs : List (GridIndex r.call.grid)) :
    ∃ s,FPSuccess.runGrid r.call (backend r) jobs (initial r heap)=some s ∧
      Integer.FullMetadata r.call s := by
  obtain ⟨s,run,memory⟩ := FPReference.whole r.call (backend r) (NativeInstructions.support r.call) jobs (initial r heap)
  have same (rg : RegionName) (ne : rg≠regions.out) (off : Nat) : s.mem rg off=(initial r heap).mem rg off := by
    rw [memory]
    exact reference_frame r jobs _ rg off ne
  have metadata := full_metadata r heap h
  refine ⟨s,run,⟨?_,?_,?_⟩,?_,?_⟩
  · intro b hb
    rw [same r.call.regions.start (by simp [Request.call,regions])]
    exact metadata.1.1 b hb
  · intro b hb
    rw [same r.call.regions.sequence (by simp [Request.call,regions])]
    exact metadata.1.2.1 b hb
  · intro b hb
    rw [same r.call.regions.prompt (by simp [Request.call,regions])]
    exact metadata.1.2.2 b hb
  · intro b hb
    rw [same r.call.regions.request (by simp [Request.call,regions])]
    exact metadata.2.1 b hb
  · intro b hb j hj
    rw [same r.call.regions.pages (by simp [Request.call,regions])]
    exact metadata.2.2 b hb j hj

/-- After every finite source-grid prefix, actual prelude/body execution reaches
all loop operands. The actual source header and signed range agree with finite
native control. The machine facts are outputs derived from the input contract. -/
theorem source_connection (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (jobs : List (GridIndex r.call.grid)) :
    ∃ s,FPSuccess.runGrid r.call (backend r) jobs (initial r heap)=some s ∧
      Integer.SourceBridge r.call (backend r) s ∧
      Integer.Header r.call (backend r) s ∧
      Integer.LoopBridge r.call Safety.widths (backend r) ∧
      Integer.Machine r.call Safety.widths := by
  obtain ⟨s,run,metadata⟩ := prefix_metadata r heap h jobs
  have finite := Safety.machine r heap h
  exact ⟨s,run,Integer.source_bridge r.call (backend r) (NativeInstructions.support r.call) s metadata,
    Integer.source_header r.call (backend r) s metadata,
    Integer.loop_bridge r.call Safety.widths finite (backend r),finite⟩

/-- The source's original min-after-prompt association and actual BM-step
backedge stay signed64-exact, including the last backedge after a reached tile. -/
theorem direct_source_control (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (idx : GridIndex r.call.grid) (step : Nat) (hs : step≤Integer.countAt r.call idx) :
    (Integer.smin
      (Integer.word ((r.call.launchAt idx).block*r.call.layout.BM)+Integer.word r.call.layout.BM+
        Integer.word (r.call.prompt (Integer.batchAt r.call idx))) (Integer.word (Integer.seqAt r.call idx))).toInt=
      (min ((r.call.launchAt idx).block*r.call.layout.BM+r.call.layout.BM+r.call.prompt (Integer.batchAt r.call idx))
        (Integer.seqAt r.call idx) : Nat) ∧
    (Integer.word (step*r.call.layout.BM)+Integer.word r.call.layout.BM).toInt=
      (step*r.call.layout.BM+r.call.layout.BM : Nat) := by
  have finite := Safety.machine r heap h
  have stop := Integer.stop_le (r.call.launchAt idx).block r.call.layout.BM (Integer.seqAt r.call idx)
    (r.call.prompt (Integer.batchAt r.call idx)) (by dsimp only [Integer.seqAt]; omega)
  have cur : step*r.call.layout.BM≤Integer.seqAt r.call idx+r.call.layout.BM := by
    rcases r.call.blockM with bm|bm <;>
      simp only [Integer.countAt,Integer.blocks,Integer.ceilTiles,bm] at hs stop ⊢ <;> split_ifs at hs <;> omega
  exact Safety.direct_control _ _ _ _ _ (Integer.block_abi r.call Safety.widths finite idx) r.call.blockM
    (finite.sequence _ (Host.batch_bound r.call idx)) (by dsimp only [Integer.seqAt]; omega) cur

#axiomsClean source_connection
#axiomsClean direct_source_control
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeSource
