import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicIOBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (withDot backwardOutputMemory)
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type}

theorem forward_memory (s : Shape) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (state : State (Expr α)) (values : TileIndex [128,s.BV] → Expr α) :
    FPStages.outputMemory algebra (config s 128) ((forwardInvocation α s tile).initial state) values=
      storeTile "o" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
        (tile.chunk*128) (tile.value*s.BV) ((forwardInvocation α s tile).initial state) values := by
  have decoded := grid_decode s 128 tile valid
  unfold FPStages.outputMemory
  change (TileShape.allIndices [128,s.BV]).foldl
    (fun t i => if tile.chunk*128+i.1.val<s.T ∧
      ProgramDomain.gridX s tile%blocks s.DV s.BV*s.BV+i.2.1.val<s.DV then
      t.write "o" ((tile.bh+s.B*s.H*(ProgramDomain.gridX s tile/blocks s.DV s.BV))*(s.T*s.DV)+
        (tile.chunk*128+i.1.val)*s.DV+(ProgramDomain.gridX s tile%blocks s.DV s.BV*s.BV+i.2.1.val)*1)
        (.mk .real (values i)) else t) ((forwardInvocation α s tile).initial state)=_
  rw [decoded.2.1,decoded.2.2]
  simp only [storeTile,tileOffset,Nat.mul_one]

theorem forward_run (s : Shape) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (state : State (Expr α)) :
    ∃ final, runSchedule [forwardInvocation α s tile] state=some final ∧
      final.mem=(storeTile "o" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
        (tile.chunk*128) (tile.value*s.BV) ((forwardInvocation α s tile).initial state)
        (FPStages.forwardValues algebra dots (config s 128) ((forwardInvocation α s tile).initial state))).mem := by
  obtain ⟨final,ran,memory⟩ := FPStages.forward_fp32 algebra dots (config s 128)
    ((forwardInvocation α s tile).initial state)
  refine ⟨final,?_,?_⟩
  · rw [runSchedule_singleton]
    exact ran
  · rw [memory,forward_memory s tile valid]
    rfl

theorem backward_run (s : Shape) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (state : State (Expr α)) :
    ∃ final, runSchedule [backwardInvocation α s tile] state=some final ∧
      final.mem=(storeTile "dv" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
        (tile.chunk*64) (tile.value*s.BV)
        (storeTile "dk" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
          (tile.chunk*64) (tile.key*s.BK)
          (storeTile "dq" ((tile.bh+s.B*s.H*tile.value)*(s.T*s.DK)) s.T s.DK
            (tile.chunk*64) (tile.key*s.BK) ((backwardInvocation α s tile).initial state)
            (FPQuery.values algebra dots (backwardConfig s tile) ((backwardInvocation α s tile).initial state)))
          (FPStages.backwardValues algebra dots (backwardConfig s tile) ((backwardInvocation α s tile).initial state)).key)
        (FPStages.backwardValues algebra dots (backwardConfig s tile) ((backwardInvocation α s tile).initial state)).value).mem := by
  let start := (backwardInvocation α s tile).initial state
  let afterQuery := FPQuery.outputMemory algebra (backwardConfig s tile) "dq" start
    (FPQuery.values algebra dots (backwardConfig s tile) start)
  have unchanged : FPStages.backwardValues algebra dots (backwardConfig s tile) afterQuery=
      FPStages.backwardValues algebra dots (backwardConfig s tile) start := by
    apply backwardValues_inputs
    all_goals apply queryMemory_inputs
    all_goals simp [backwardConfig,config]
  obtain ⟨final,ran,memory⟩ := FPBackward.backward_fp32 algebra dots (backwardConfig s tile) "dq" start
  refine ⟨final,?_,?_⟩
  · rw [runSchedule_singleton]
    unfold Invocation.run
    rw [backward_source s tile valid]
    exact ran
  · change final.mem=(backwardOutputMemory algebra (backwardConfig s tile) afterQuery
        (FPStages.backwardValues algebra dots (backwardConfig s tile) afterQuery)).mem at memory
    rw [unchanged,backwardMemory_eq] at memory
    simpa only [afterQuery,start,queryMemory_eq] using memory

/-- Full cell frame, including inactive lanes inside an output region. -/
theorem storeTile_frame_full {β : Type} (region other : RegionName)
    (base rows cols rowOffset colOffset : Nat) (state : State β)
    (values : TileIndex [h,w] → β) (offset : Nat)
    (outside : ∀ i : TileIndex [h,w],
      rowOffset+i.1.val<rows ∧ colOffset+i.2.1.val<cols →
      other≠region ∨ offset≠tileOffset base cols rowOffset colOffset i) :
    (storeTile region base rows cols rowOffset colOffset state values).mem other offset=state.mem other offset := by
  unfold storeTile
  apply State.masked_scatter_frame
  by_cases different : other≠region
  · exact Or.inl different
  · right
    intro i member active
    exact (outside i active).resolve_left different

#axiomsClean forward_run
#axiomsClean backward_run
#axiomsClean storeTile_frame_full
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
