import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedCoverage

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem time_outputWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (domain : StoreDomain s b) (r : RegionName) (offset : Nat) :
    covers (timeWrites s.T (outputWrites a s b plans initial seed pid) 0) r offset ↔
      r="o" ∧ ∃ t,t<s.T ∧ ∃ lane,lane<s.BV ∧ pid.value s lane<s.DV ∧
        offset=partialOffset s s.DV pid.ik pid.head t (pid.value s lane) := by
  rw [covers_timeWrites]
  constructor
  · rintro ⟨n,_,hn,hc⟩
    rcases (covers_outputWrites a s b plans initial seed final pid n domain hn r offset).mp hc with
      ⟨hr,lane,hl,hm,he⟩
    exact ⟨hr,n,by omega,lane,hl,hm,he⟩
  · rintro ⟨hr,t,ht,lane,hl,hm,he⟩
    refine ⟨t,by omega,by omega,?_⟩
    apply (covers_outputWrites a s b plans initial seed final pid _ domain (by omega) r offset).mpr
    refine ⟨hr,lane,hl,hm,?_⟩
    exact he

theorem time_queryWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (domain : StoreDomain s b) (r : RegionName) (offset : Nat) :
    covers (timeWrites s.T (queryWrites a s b plans initial seed pid) 0) r offset ↔
      r="dq" ∧ ∃ t,t<s.T ∧ ∃ lane,lane<s.BK ∧ pid.key s lane<s.DK ∧
        offset=partialOffset s s.DK pid.iv pid.head t (pid.key s lane) := by
  rw [covers_timeWrites]
  constructor
  · rintro ⟨n,_,hn,hc⟩
    rcases (covers_queryWrites a s b plans initial seed final pid n domain hn r offset).mp hc with
      ⟨hr,lane,hl,hm,he⟩
    exact ⟨hr,n,by omega,lane,hl,hm,he⟩
  · rintro ⟨hr,t,ht,lane,hl,hm,he⟩
    refine ⟨t,by omega,by omega,?_⟩
    apply (covers_queryWrites a s b plans initial seed final pid _ domain (by omega) r offset).mpr
    refine ⟨hr,lane,hl,hm,?_⟩
    exact he

theorem time_reverseKeyWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (domain : StoreDomain s b) (r : RegionName) (offset : Nat) :
    covers (timeWrites s.T (reverseKeyWrites a s b plans initial final pid) 0) r offset ↔
      r="dk" ∧ ∃ t,t<s.T ∧ ∃ lane,lane<s.BK ∧ pid.key s lane<s.DK ∧
        offset=partialOffset s s.DK pid.iv pid.head t (pid.key s lane) := by
  rw [covers_timeWrites]
  constructor
  · rintro ⟨n,_,hn,hc⟩
    rcases (covers_reverseKeyWrites a s b plans initial seed final pid n domain hn r offset).mp hc with
      ⟨hr,lane,hl,hm,he⟩
    exact ⟨hr,s.T-1-n,by omega,lane,hl,hm,he⟩
  · rintro ⟨hr,t,ht,lane,hl,hm,he⟩
    refine ⟨s.T-1-t,by omega,by omega,?_⟩
    apply (covers_reverseKeyWrites a s b plans initial seed final pid _ domain (by omega) r offset).mpr
    refine ⟨hr,lane,hl,hm,?_⟩
    simpa only [show s.T-1-(s.T-1-t)=t by omega] using he

theorem time_reverseValueWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (domain : StoreDomain s b) (r : RegionName) (offset : Nat) :
    covers (timeWrites s.T (reverseValueWrites a s b plans initial final pid) 0) r offset ↔
      r="dv" ∧ ∃ t,t<s.T ∧ ∃ lane,lane<s.BV ∧ pid.value s lane<s.DV ∧
        offset=partialOffset s s.DV pid.ik pid.head t (pid.value s lane) := by
  rw [covers_timeWrites]
  constructor
  · rintro ⟨n,_,hn,hc⟩
    rcases (covers_reverseValueWrites a s b plans initial seed final pid n domain hn r offset).mp hc with
      ⟨hr,lane,hl,hm,he⟩
    exact ⟨hr,s.T-1-n,by omega,lane,hl,hm,he⟩
  · rintro ⟨hr,t,ht,lane,hl,hm,he⟩
    refine ⟨s.T-1-t,by omega,by omega,?_⟩
    apply (covers_reverseValueWrites a s b plans initial seed final pid _ domain (by omega) r offset).mpr
    refine ⟨hr,lane,hl,hm,?_⟩
    simpa only [show s.T-1-(s.T-1-t)=t by omega] using he

theorem time_reverseWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : RegionName) (offset : Nat) :
    covers (timeWrites s.T (reverseWrites a s b plans initial final pid) 0) r offset ↔
      covers (timeWrites s.T (reverseKeyWrites a s b plans initial final pid) 0) r offset ∨
      covers (timeWrites s.T (reverseValueWrites a s b plans initial final pid) 0) r offset := by
  simp only [covers_timeWrites,reverseWrites,covers_append]
  constructor
  · rintro ⟨t,ht,ht',h|h⟩
    · exact Or.inl ⟨t,ht,ht',h⟩
    · exact Or.inr ⟨t,ht,ht',h⟩
  · rintro (⟨t,ht,ht',h⟩|⟨t,ht,ht',h⟩)
    · exact ⟨t,ht,ht',Or.inl h⟩
    · exact ⟨t,ht,ht',Or.inr h⟩

def forwardRange (s : Shape) (final : Bool) (r : RegionName) (i : Nat) : Prop :=
  (r="o" ∧ i<s.NK*rowCount s s.DV) ∨ (final=Bool.true ∧ r="final_state" ∧ i<s.BH*s.DK*s.DV)

def backwardRange (s : Shape) (r : RegionName) (i : Nat) : Prop :=
  (r="dq" ∧ i<s.NV*rowCount s s.DK) ∨ (r="dk" ∧ i<s.NV*rowCount s s.DK) ∨
    (r="dv" ∧ i<s.NK*rowCount s s.DV)

def writtenRange (mode : Mode) (s : Shape) (final : Bool) (r : RegionName) (i : Nat) : Prop :=
  match mode with
  | .forward => forwardRange s final r i
  | .backward => backwardRange s r i

theorem forward_coverage [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (domain : StoreDomain s b)
    (r : RegionName) (i : Nat) :
    covers (gridStores .forward a s b plans initial seed final (gridPrograms s)) r i ↔ forwardRange s final r i := by
  simp only [gridStores,covers_flatMap,mem_gridPrograms,programStores,forwardStores,covers_append,
    time_outputWrites a s b plans initial seed final _ domain,covers_finalWrites]
  constructor
  · rintro ⟨pid,hp,ho|hf⟩
    · rcases ho with ⟨hr,t,ht,lane,_,hl,rfl⟩
      exact Or.inl ⟨hr,partialOffset_bound s s.NK s.DV _ _ _ _ hp.2.1 hp.2.2 ht hl⟩
    · rcases hf with ⟨hf,hr,lk,_,lv,_,hk,hv,rfl⟩
      exact Or.inr ⟨hf,hr,stateOffset_bound s _ _ _ hp.2.2 hk hv⟩
  · rintro (⟨hr,hi⟩|⟨hf,hr,hi⟩)
    · obtain ⟨hs,hh,ht,hv,he⟩ := unflatten_partial s s.NK s.DV i hi
      obtain ⟨hp,hl,hval⟩ := output_owner s domain.shape _ _ _ hs hh hv
      refine ⟨_,hp,Or.inl ⟨hr,_,ht,_,hl,?_,?_⟩⟩
      · simpa only [hval] using hv
      · rw [hval]; exact he.symm
    · obtain ⟨hh,hk,hv,he⟩ := unflatten_state s i hi
      obtain ⟨hp,hlk,hlv,hkey,hval⟩ := final_owner s domain.shape _ _ _ hh hk hv
      refine ⟨_,hp,Or.inr ⟨hf,hr,_,hlk,_,hlv,?_,?_,?_⟩⟩
      · simpa only [hkey] using hk
      · simpa only [hval] using hv
      · rw [hkey,hval]; exact he.symm

theorem backward_coverage [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (domain : StoreDomain s b)
    (r : RegionName) (i : Nat) :
    covers (gridStores .backward a s b plans initial seed final (gridPrograms s)) r i ↔ backwardRange s r i := by
  simp only [gridStores,covers_flatMap,mem_gridPrograms,programStores,backwardStores,covers_append,
    time_queryWrites a s b plans initial seed final _ domain,time_reverseWrites,
    time_reverseKeyWrites a s b plans initial seed final _ domain,
    time_reverseValueWrites a s b plans initial seed final _ domain]
  constructor
  · rintro ⟨pid,hp,hq|hk|hv⟩
    · rcases hq with ⟨hr,t,ht,lane,_,hl,rfl⟩
      exact Or.inl ⟨hr,partialOffset_bound s s.NV s.DK _ _ _ _ hp.1 hp.2.2 ht hl⟩
    · rcases hk with ⟨hr,t,ht,lane,_,hl,rfl⟩
      exact Or.inr (Or.inl ⟨hr,partialOffset_bound s s.NV s.DK _ _ _ _ hp.1 hp.2.2 ht hl⟩)
    · rcases hv with ⟨hr,t,ht,lane,_,hl,rfl⟩
      exact Or.inr (Or.inr ⟨hr,partialOffset_bound s s.NK s.DV _ _ _ _ hp.2.1 hp.2.2 ht hl⟩)
  · rintro (⟨hr,hi⟩|⟨hr,hi⟩|⟨hr,hi⟩)
    · obtain ⟨hs,hh,ht,hk,he⟩ := unflatten_partial s s.NV s.DK i hi
      obtain ⟨hp,hl,hkey⟩ := gradient_owner s domain.shape _ _ _ hs hh hk
      refine ⟨_,hp,Or.inl ⟨hr,_,ht,_,hl,?_,?_⟩⟩
      · simpa only [hkey] using hk
      · rw [hkey]; exact he.symm
    · obtain ⟨hs,hh,ht,hk,he⟩ := unflatten_partial s s.NV s.DK i hi
      obtain ⟨hp,hl,hkey⟩ := gradient_owner s domain.shape _ _ _ hs hh hk
      refine ⟨_,hp,Or.inr (Or.inl ⟨hr,_,ht,_,hl,?_,?_⟩)⟩
      · simpa only [hkey] using hk
      · rw [hkey]; exact he.symm
    · obtain ⟨hs,hh,ht,hv,he⟩ := unflatten_partial s s.NK s.DV i hi
      obtain ⟨hp,hl,hval⟩ := output_owner s domain.shape _ _ _ hs hh hv
      refine ⟨_,hp,Or.inr (Or.inr ⟨hr,_,ht,_,hl,?_,?_⟩)⟩
      · simpa only [hval] using hv
      · rw [hval]; exact he.symm

#axiomsClean forward_coverage
#axiomsClean backward_coverage
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
