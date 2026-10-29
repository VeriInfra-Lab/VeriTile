import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopState

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealCarry
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The real source's completion/reset and continuation branches preserve the
flat-loop register invariant. A completed tile resets before the next tile's
coordinates are installed; a continuing tile retains the accumulated sum. -/
theorem preserves (p : Parameters) (s : BlockState) (xs : RealMath.Inputs) (pid i : Nat)
    (hk : 0<count p) (hf : Fixed p s) (hc : Coordinates p pid i s)
    (hi : s.regs .int [] "ki" = some (Tile.scalar (phase p i : Int)))
    (ht : s.regs .int [] "tile_id" = some (Tile.scalar (tile p pid i : Int))) :
    Holds p xs pid (i+1)
      (RealStep.result p s xs (row p pid i) (column p pid i) (phase p i)) := by
  have count_eq : RealMath.blocks p.K p.BK = count p := rfl
  have hsame (h : phase p i+1≠count p) :
      row p pid (i+1)=row p pid i ∧ column p pid (i+1)=column p pid i := by
    simp [row,column,continue_tile p pid i hk h]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · constructor <;> by_cases hlast : phase p i+1=count p <;>
      simp [RealStep.result,count_eq,hlast,RealStep.middle,RealLoads.result,
        RealStore.result,RealStore.write,RealStore.prepared,hf.count,hf.rows,hf.group,hf.lanes]
  · by_cases hlast : phase p i+1=count p <;>
      simp [RealStep.result,count_eq,hlast,RealStep.middle,RealLoads.result,
        RealStore.result,RealStore.write,RealStore.prepared,previousK,hi]
    all_goals simp [phase]
  · rw [next_tile_register]
    by_cases hlast : phase p i+1=count p <;>
      simp [RealStep.result,count_eq,hlast,RealStep.middle,RealLoads.result,
        RealStore.result,RealStore.write,RealStore.prepared,ht]
  · by_cases hlast : phase p i+1=count p
    · have hn : phase p (i+1)=0 := (FlatSchedule.last_iff_restart (count p) i hk).mp hlast
      simp [RealStep.result,count_eq,hlast,RealStore.result,accumulator,hn,
        RealDot.accumulator,RealMath.partial_zero]
    · have hn := continue_phase p i hk hlast
      have ht' := hsame hlast
      simp [RealStep.result,count_eq,hlast,RealStep.middle,accumulator,hn,ht'.1,ht'.2]
  · intro hn
    have hlast : phase p i+1≠count p := by
      intro h
      exact hn ((FlatSchedule.last_iff_restart (count p) i hk).mp h)
    have ht' := hsame hlast
    constructor <;>
      simp [RealStep.result,count_eq,hlast,RealStep.middle,RealLoads.result,
        ht'.1,ht'.2,hc.row,hc.column,hc.a,hc.b]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealCarry
