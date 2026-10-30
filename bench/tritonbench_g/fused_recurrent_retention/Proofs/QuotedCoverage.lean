import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedGridExecution

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem covers_flatMap {ι : Type} (items : List ι) (f : ι → List (Write α)) (r : RegionName) (i : Nat) :
    covers (items.flatMap f) r i ↔ ∃ j∈items,covers (f j) r i := by
  induction items with
  | nil => simp [covers]
  | cons j rest ih => simp [covers_append,ih]

theorem unflatten_partial (s : Shape) (count width i : Nat) (hi : i<count*rowCount s width) :
    flatSplit s width i<count ∧ flatHead s width i<s.BH ∧ flatTime s width i<s.T ∧
    flatChannel width i<width ∧
    partialOffset s width (flatSplit s width i) (flatHead s width i)
      (flatTime s width i) (flatChannel width i)=i := by
  have hw : 0<width := by
    by_contra h
    have hz : width=0 := by omega
    simp [rowCount,hz] at hi
  have ht : 0<s.T := by
    by_contra h
    have hz : s.T=0 := by omega
    simp [rowCount,hz] at hi
  have hh : 0<s.BH := by
    by_contra h
    have hz : s.BH=0 := by omega
    simp [rowCount,hz] at hi
  have hd : i/width/s.T/s.BH<count := by
    apply (Nat.div_lt_iff_lt_mul hh).mpr
    apply (Nat.div_lt_iff_lt_mul ht).mpr
    apply (Nat.div_lt_iff_lt_mul hw).mpr
    simpa only [rowCount,Nat.mul_assoc] using hi
  refine ⟨hd,Nat.mod_lt _ hh,Nat.mod_lt _ ht,Nat.mod_lt _ hw,?_⟩
  unfold partialOffset flatSplit flatHead flatTime flatChannel
  rw [owner_reconstruct,owner_reconstruct,owner_reconstruct]

theorem unflatten_state (s : Shape) (i : Nat) (hi : i<s.BH*s.DK*s.DV) :
    i/s.DV/s.DK<s.BH ∧ i/s.DV%s.DK<s.DK ∧ i%s.DV<s.DV ∧
    stateOffset s (i/s.DV/s.DK) (i/s.DV%s.DK) (i%s.DV)=i := by
  have hv : 0<s.DV := by
    by_contra h
    have hz : s.DV=0 := by omega
    simp [hz] at hi
  have hk : 0<s.DK := by
    by_contra h
    have hz : s.DK=0 := by omega
    simp [hz] at hi
  refine ⟨?_,Nat.mod_lt _ hk,Nat.mod_lt _ hv,?_⟩
  · apply (Nat.div_lt_iff_lt_mul hk).mpr
    apply (Nat.div_lt_iff_lt_mul hv).mpr
    exact hi
  · unfold stateOffset
    rw [owner_reconstruct,owner_reconstruct]

theorem covers_outputWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (ht : t<s.T) (r : RegionName) (offset : Nat) :
    covers (outputWrites a s b plans initial seed pid t) r offset ↔
      r="o" ∧ ∃ lane,lane<s.BV ∧ pid.value s lane<s.DV ∧
        offset=partialOffset s s.DV pid.ik pid.head (t) (pid.value s lane) := by
  unfold outputWrites
  rw [covers_filterMap]
  constructor
  · rintro ⟨i,_,hi,hr,he⟩
    refine ⟨hr,i.1.val,i.1.isLt,hi,?_⟩
    rw [he]
    exact outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1 pid _ _
  · rintro ⟨hr,lane,hl,hi,he⟩
    refine ⟨(⟨lane,hl⟩,PUnit.unit),TileShape.mem_allIndices _ _,hi,hr,?_⟩
    rw [he]
    exact (outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1 pid _ _).symm

theorem covers_queryWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (ht : t<s.T) (r : RegionName) (offset : Nat) :
    covers (queryWrites a s b plans initial seed pid t) r offset ↔
      r="dq" ∧ ∃ lane,lane<s.BK ∧ pid.key s lane<s.DK ∧
        offset=partialOffset s s.DK pid.iv pid.head (t) (pid.key s lane) := by
  unfold queryWrites
  rw [covers_filterMap]
  constructor
  · rintro ⟨i,_,hi,hr,he⟩
    refine ⟨hr,i.1.val,i.1.isLt,hi,?_⟩
    rw [he]
    exact gradientOffset_partial s b domain.qDense (by omega) domain.shape.1 pid _ _
  · rintro ⟨hr,lane,hl,hi,he⟩
    refine ⟨(⟨lane,hl⟩,PUnit.unit),TileShape.mem_allIndices _ _,hi,hr,?_⟩
    rw [he]
    exact (gradientOffset_partial s b domain.qDense (by omega) domain.shape.1 pid _ _).symm

theorem covers_reverseKeyWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (ht : t<s.T) (r : RegionName) (offset : Nat) :
    covers (reverseKeyWrites a s b plans initial final pid t) r offset ↔
      r="dk" ∧ ∃ lane,lane<s.BK ∧ pid.key s lane<s.DK ∧
        offset=partialOffset s s.DK pid.iv pid.head (s.T-1-t) (pid.key s lane) := by
  unfold reverseKeyWrites
  rw [covers_filterMap]
  constructor
  · rintro ⟨i,_,hi,hr,he⟩
    refine ⟨hr,i.1.val,i.1.isLt,hi,?_⟩
    rw [he]
    rw [reverseGradient_at _ _ _ _ ht]
    exact gradientOffset_partial s b domain.qDense (by omega) domain.shape.1 pid _ _
  · rintro ⟨hr,lane,hl,hi,he⟩
    refine ⟨(⟨lane,hl⟩,PUnit.unit),TileShape.mem_allIndices _ _,hi,hr,?_⟩
    rw [he]
    rw [reverseGradient_at _ _ _ _ ht]
    exact (gradientOffset_partial s b domain.qDense (by omega) domain.shape.1 pid _ _).symm

theorem covers_reverseValueWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (ht : t<s.T) (r : RegionName) (offset : Nat) :
    covers (reverseValueWrites a s b plans initial final pid t) r offset ↔
      r="dv" ∧ ∃ lane,lane<s.BV ∧ pid.value s lane<s.DV ∧
        offset=partialOffset s s.DV pid.ik pid.head (s.T-1-t) (pid.value s lane) := by
  unfold reverseValueWrites
  rw [covers_filterMap]
  constructor
  · rintro ⟨i,_,hi,hr,he⟩
    refine ⟨hr,i.1.val,i.1.isLt,hi,?_⟩
    rw [he]
    rw [reverseOutput_at _ _ _ _ ht]
    exact outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1 pid _ _
  · rintro ⟨hr,lane,hl,hi,he⟩
    refine ⟨(⟨lane,hl⟩,PUnit.unit),TileShape.mem_allIndices _ _,hi,hr,?_⟩
    rw [he]
    rw [reverseOutput_at _ _ _ _ ht]
    exact (outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1 pid _ _).symm

theorem covers_finalWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (r : RegionName) (offset : Nat) :
    covers (finalWrites a s b initial seed final pid) r offset ↔
      final=Bool.true ∧ r="final_state" ∧ ∃ lk,lk<s.BK ∧ ∃ lv,lv<s.BV ∧
        pid.key s lk<s.DK ∧ pid.value s lv<s.DV ∧ offset=stateOffset s pid.head (pid.key s lk) (pid.value s lv) := by
  cases final
  · simp [finalWrites,covers]
  · unfold finalWrites
    simp only [reduceIte,true_and]
    rw [covers_filterMap]
    constructor
    · rintro ⟨i,_,⟨hk,hv⟩,hr,he⟩
      exact ⟨hr,i.2.1.val,i.2.1.isLt,i.1.val,i.1.isLt,hk,hv,he.trans (seedOffset_state _ _ _ _ _)⟩
    · rintro ⟨hr,lk,hlk,lv,hlv,hk,hv,he⟩
      exact ⟨(⟨lv,hlv⟩,⟨lk,hlk⟩,PUnit.unit),TileShape.mem_allIndices _ _,⟨hk,hv⟩,hr,he.trans (seedOffset_state _ _ _ _ _).symm⟩

#axiomsClean unflatten_partial
#axiomsClean covers_outputWrites
#axiomsClean covers_finalWrites
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
