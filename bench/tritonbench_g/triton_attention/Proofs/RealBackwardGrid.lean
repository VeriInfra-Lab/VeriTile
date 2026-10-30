import bench.tritonbench_g.triton_attention.Proofs.RealBackwardReference
import bench.tritonbench_g.triton_attention.Proofs.GridUnique
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def HeadZero (c : Shape) (head : Fin c.groups) (mem : Grid.Memory R) : Prop :=
  ∀ i : Coord c,i.1=head → mem "DQ" (offset c i)=.mk .fp32 (some 0)

theorem backward_separate (c : Shape) (scale : ℝ) : BackwardSeparate (backwardParameters c scale) := by
  constructor <;> simp [backwardParameters]

theorem backwardStep_reads (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads (backwardParameters c scale) mem μ) :
    BackwardReads (backwardParameters c scale) (Grid.backwardStep model c scale pid mem) μ := by
  constructor
  · intro r hr o ho
    rw [Grid.backwardStep_readonly model c scale pid mem r o]
    · exact h.1 r hr o ho
    · simp only [backwardParameters,List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl <;> decide
  · intro r hr o ho
    rw [Grid.backwardStep_readonly model c scale pid mem r o]
    · exact h.2 r hr o ho
    · simp only [backwardParameters,List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl <;> decide

theorem backwardStep_headZero (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (hp : Grid.Legal (Grid.backwardGrid c) pid) (target : Coord c) (hne : pid≠Grid.backwardProgram c target)
    (h : HeadZero c target.1 mem) : HeadZero c target.1 (Grid.backwardStep model c scale pid mem) := by
  intro i hi
  rw [Grid.backwardStep_other model c scale pid mem hp i (by simpa [Grid.backwardProgram,hi] using hne) "DQ"]
  exact h i hi

theorem backwardStep_values (c : Shape) (scale : ℝ) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads (backwardParameters c scale) mem μ)
    (i : Coord c) (hz : HeadZero c i.1 mem) :
    Grid.backwardStep model c scale (Grid.backwardProgram c i) mem "DQ" (offset c i)=.mk .fp32 (some (tiledQ c scale μ i)) ∧
    Grid.backwardStep model c scale (Grid.backwardProgram c i) mem "DK" (offset c i)=.mk .fp32 (some (tiledK c scale μ i)) ∧
    Grid.backwardStep model c scale (Grid.backwardProgram c i) mem "DV" (offset c i)=.mk .fp32 (some (tiledV c scale μ i)) := by
  let pid:=Grid.backwardProgram c i
  let s:=Grid.entry (Grid.backwardGrid c) pid mem
  have hp:=Grid.backwardProgram_legal c i
  have hb : ∀ b<(backwardParameters c scale).blocks,∀ j : Fin 128,
      BackwardRowBound (backwardParameters c scale) s b j := by
    intro b hbn j
    constructor
    · dsimp only [s]
      rw [Grid.backwardHead_eq]
      exact Grid.row_bound c pid.1 b j.val hp.1 hbn j.isLt
    · simpa [backwardRowOffset,backwardParameters,s,Grid.entry,Grid.ids,Shape.rows,Nat.add_assoc,Nat.mul_comm]
        using Grid.row_bound c pid.1 b j.val hp.1 hbn j.isLt
  have hq : ∀ b,b<(backwardParameters c scale).blocks → ∀ lane : BLane (backwardParameters c scale),
      s.mem (backwardParameters c scale).dq (backwardOffset (backwardParameters c scale) s b lane)=.mk .fp32 (some 0) := by
    intro b hbn lane
    have hr : b*128+lane.1.val<c.rows := by
      have hl:=lane.1.isLt
      change lane.1.val<128 at hl
      change b<c.blocks at hbn
      change _<c.blocks*128
      omega
    have hh:=hz (i.1,⟨b*128+lane.1.val,hr⟩,lane.2.1) rfl
    dsimp only [s]
    rw [Grid.backwardOffset_eq]
    simpa [offset,pid,Grid.backwardProgram,Nat.add_assoc] using hh
  obtain ⟨final,he,ho,_⟩:=backward_execution_real (backwardParameters c scale) s μ h
    (backward_separate c scale) (by change 0<128;decide) hb hq
  change FP.Structural.exec (backend algebra model) (backwardKernel c scale)
    (Grid.entry (Grid.backwardGrid c) pid mem)=some final at he
  have hm : final.mem=Grid.backwardStep model c scale pid mem := by
    simpa only [Grid.execute,he,Option.map_some,Option.some.injEq] using Grid.backward_success algebra model c scale pid mem
  have ans:=ho (i.2.1.val/128) (grid_coverage c i.2.1).1 (Grid.backwardLane c scale i)
  rw [hm] at ans
  dsimp only [s,pid] at ans
  rw [Grid.backward_target_offset] at ans
  simp only [queryContribution_entry,keyContribution_entry,valueContribution_entry] at ans
  simpa [tiledQ,tiledK,tiledV,Grid.backwardProgram,Grid.backwardLane,backwardParameters,ofFloat,Nat.div_add_mod'] using ans

theorem backward_grid_values (c : Shape) (scale : ℝ) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.backwardGrid c) work) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads (backwardParameters c scale) mem μ)
    (hz : ∀ i : Coord c,mem "DQ" (offset c i)=.mk .fp32 (some 0)) :
    ∃ final,Grid.run (Grid.execute algebra model (backwardKernel c scale) (Grid.backwardGrid c)) work mem=some final ∧
      BackwardReads (backwardParameters c scale) final μ ∧
      ∀ i : Coord c,final "DQ" (offset c i)=.mk .fp32 (some (tiledQ c scale μ i)) ∧
        final "DK" (offset c i)=.mk .fp32 (some (tiledK c scale μ i)) ∧
        final "DV" (offset c i)=.mk .fp32 (some (tiledV c scale μ i)) := by
  let final:=work.foldl (fun m pid=>Grid.backwardStep model c scale pid m) mem
  have hlegal:=Grid.FullSchedule.legal (Grid.backwardGrid c) work schedule
  have hnodup:=Grid.schedule_nodup (Grid.backwardGrid c) work schedule
  have hinv : BackwardReads (backwardParameters c scale) final μ :=
    Grid.fold_invariant (Grid.backwardStep model c scale) (fun m=>BackwardReads (backwardParameters c scale) m μ) work mem h
      (by intro pid hp m hm;exact backwardStep_reads c scale pid m μ hm)
  refine ⟨final,Grid.run_eq_fold _ _ (Grid.backward_success algebra model c scale) work mem,hinv,?_⟩
  intro i
  have aux (r : RegionName) (value : ℝ)
      (hw : ∀ m,BackwardReads (backwardParameters c scale) m μ → HeadZero c i.1 m →
        Grid.backwardStep model c scale (Grid.backwardProgram c i) m r (offset c i)=.mk .fp32 (some value)) :
      final r (offset c i)=.mk .fp32 (some value) := by
    apply Grid.fold_readback_unique (Grid.backwardStep model c scale)
      (fun m=>BackwardReads (backwardParameters c scale) m μ ∧ HeadZero c i.1 m)
      work mem (Grid.backwardProgram c i) r (offset c i) _ hnodup
      (Grid.FullSchedule.covers _ work schedule _ (Grid.backwardProgram_legal c i)) ⟨h,by intro j hj;exact hz j⟩
    · intro pid hp hn m hm
      exact ⟨backwardStep_reads c scale pid m μ hm.1,backwardStep_headZero c scale pid m (hlegal pid hp) i hn hm.2⟩
    · intro m hm
      exact hw m hm.1 hm.2
    · intro pid hp hn m
      exact Grid.backwardStep_other model c scale pid m (hlegal pid hp) i hn r
  exact ⟨aux "DQ" _ (by intro m hm hzero;exact (backwardStep_values c scale m μ hm i hzero).1),
    aux "DK" _ (by intro m hm hzero;exact (backwardStep_values c scale m μ hm i hzero).2.1),
    aux "DV" _ (by intro m hm hzero;exact (backwardStep_values c scale m μ hm i hzero).2.2)⟩

#axiomsClean backward_grid_values
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
