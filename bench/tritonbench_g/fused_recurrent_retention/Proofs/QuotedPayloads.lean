import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedStoreGeometry

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Flat payload lookup, proved below equal to every actual store. -/
def payload (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (r : RegionName) (i : Nat) : Cell (Staged.Node α) :=
  .mk .real (Staged.atom (if r="o" then sourceCell a s plans x seed final .output i
    else if r="final_state" then sourceCell a s plans x seed final .finalState i
    else if r="dq" then sourceCell a s plans x seed final .dq i
    else if r="dk" then sourceCell a s plans x seed final .dk i
    else if r="dv" then sourceCell a s plans x seed final .dv i else a.zero))

theorem cell_O (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (pid : ProgramId) (t lane : Nat) (hp : pid.Valid s)
    (ht : t<s.T) (hl : pid.value s lane<s.DV) :
    sourceCell a s plans x seed final .output (partialOffset s s.DV pid.ik pid.head t (pid.value s lane))=
      programO a s plans x seed pid t lane := by
  obtain ⟨hs,hh,htt,hc⟩ := flat_partial s s.DV pid.ik pid.head t (pid.value s lane) hp.2.2 ht hl
  rw [sourceCell_partial,programO_partial]
  simp [partialValue,PartialKind.width,hs,hh,htt,hc]

theorem cell_Q (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (pid : ProgramId) (t lane : Nat) (hp : pid.Valid s)
    (ht : t<s.T) (hl : pid.key s lane<s.DK) :
    sourceCell a s plans x seed final .dq (partialOffset s s.DK pid.iv pid.head t (pid.key s lane))=
      programQ a s plans x seed pid t lane := by
  obtain ⟨hs,hh,htt,hc⟩ := flat_partial s s.DK pid.iv pid.head t (pid.key s lane) hp.2.2 ht hl
  rw [sourceCell_partial,programQ_partial]
  simp [partialValue,PartialKind.width,hs,hh,htt,hc]

theorem cell_K (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (pid : ProgramId) (t lane : Nat) (hp : pid.Valid s)
    (ht : t<s.T) (hl : pid.key s lane<s.DK) :
    sourceCell a s plans x seed final .dk (partialOffset s s.DK pid.iv pid.head t (pid.key s lane))=
      programK a s plans x final pid t lane := by
  obtain ⟨hs,hh,htt,hc⟩ := flat_partial s s.DK pid.iv pid.head t (pid.key s lane) hp.2.2 ht hl
  rw [sourceCell_partial,programK_partial]
  simp [partialValue,PartialKind.width,hs,hh,htt,hc]

theorem cell_V (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (pid : ProgramId) (t lane : Nat) (hp : pid.Valid s)
    (ht : t<s.T) (hl : pid.value s lane<s.DV) :
    sourceCell a s plans x seed final .dv (partialOffset s s.DV pid.ik pid.head t (pid.value s lane))=
      programV a s plans x final pid t lane := by
  obtain ⟨hs,hh,htt,hc⟩ := flat_partial s s.DV pid.ik pid.head t (pid.value s lane) hp.2.2 ht hl
  rw [sourceCell_partial,programV_partial]
  simp [partialValue,PartialKind.width,hs,hh,htt,hc]

theorem cell_final (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α)
    (seed final : Bool) (pid : ProgramId) (lk lv : Nat)
    (hk : pid.key s lk<s.DK) (hv : pid.value s lv<s.DV) :
    sourceCell a s plans x seed final .finalState (stateOffset s pid.head (pid.key s lk) (pid.value s lv))=
      programForwardCarry a s x seed pid s.T lk lv := by
  rw [sourceCell_partial,programForwardCarry_at]
  simp only [partialValue,stateOffset,pair_div _ _ _ hv,pair_div _ _ _ hk,
    pair_mod _ _ _ hv,pair_mod _ _ _ hk]

theorem outputWrites_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (hp : pid.Valid s) (ht : t<s.T)
    (w : Write (Staged.Node α)) (hw : w∈outputWrites a s b plans initial seed pid t) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,hi,rfl⟩
  simp only [payload]
  simp only [Region.mk.injEq,String.reduceEq,reduceIte]
  rw [outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1]
  rw [cell_O a s plans _ seed final pid _ _ hp (by exact ht) hi]

theorem queryWrites_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (hp : pid.Valid s) (ht : t<s.T)
    (w : Write (Staged.Node α)) (hw : w∈queryWrites a s b plans initial seed pid t) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,hi,rfl⟩
  simp only [payload]
  simp only [Region.mk.injEq,String.reduceEq,reduceIte]
  rw [gradientOffset_partial s b domain.qDense (by omega) domain.shape.1]
  rw [cell_Q a s plans _ seed final pid _ _ hp (by exact ht) hi]

theorem reverseKeyWrites_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (hp : pid.Valid s) (ht : t<s.T)
    (w : Write (Staged.Node α)) (hw : w∈reverseKeyWrites a s b plans initial final pid t) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,hi,rfl⟩
  simp only [payload]
  simp only [Region.mk.injEq,String.reduceEq,reduceIte]
  rw [reverseGradient_at _ _ _ _ ht]
  rw [gradientOffset_partial s b domain.qDense (by omega) domain.shape.1]
  rw [cell_K a s plans _ seed final pid _ _ hp (by dsimp [layout]; omega) hi]
  rfl

theorem reverseValueWrites_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) (t : Nat)
    (domain : StoreDomain s b) (hp : pid.Valid s) (ht : t<s.T)
    (w : Write (Staged.Node α)) (hw : w∈reverseValueWrites a s b plans initial final pid t) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,hi,rfl⟩
  simp only [payload]
  simp only [Region.mk.injEq,String.reduceEq,reduceIte]
  rw [reverseOutput_at _ _ _ _ ht]
  rw [outputOffset_partial s b domain.vDense (by omega) domain.shape.2.1]
  rw [cell_V a s plans _ seed final pid _ _ hp (by dsimp [layout]; omega) hi]
  rfl

theorem finalWrites_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (w : Write (Staged.Node α)) (hw : w∈finalWrites a s b initial seed final pid) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  cases final
  · simp [finalWrites] at hw
  · rcases filtered_member _ _ _ _ hw with ⟨i,_,hi,rfl⟩
    simp only [payload]
    simp only [Region.mk.injEq,String.reduceEq,reduceIte]
    rw [seedOffset_state,cell_final a s plans _ seed Bool.true pid _ _ hi.1 hi.2]

theorem forwardStores_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (domain : StoreDomain s b) (hp : pid.Valid s)
    (w : Write (Staged.Node α)) (hw : w∈forwardStores a s b plans initial seed final pid) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  rcases List.mem_append.mp hw with h|h
  · rcases (mem_timeWrites _ _ _ _).mp h with ⟨t,_,ht,h⟩
    exact outputWrites_payload _ _ _ _ _ _ _ _ _ domain hp ht w h
  · exact finalWrites_payload _ _ _ _ _ _ _ _ w h

theorem backwardStores_payload [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (domain : StoreDomain s b) (hp : pid.Valid s)
    (w : Write (Staged.Node α)) (hw : w∈backwardStores a s b plans initial seed final pid) :
    w.value=payload a s plans (inputs (layout s b) initial) seed final w.region w.offset := by
  rcases List.mem_append.mp hw with h|h
  · rcases (mem_timeWrites _ _ _ _).mp h with ⟨t,_,ht,h⟩
    exact queryWrites_payload _ _ _ _ _ _ _ _ _ domain hp ht w h
  · rcases (mem_timeWrites _ _ _ _).mp h with ⟨t,_,ht,h⟩
    rcases List.mem_append.mp h with h|h
    · exact reverseKeyWrites_payload _ _ _ _ _ _ _ _ _ domain hp ht w h
    · exact reverseValueWrites_payload _ _ _ _ _ _ _ _ _ domain hp ht w h

#axiomsClean forwardStores_payload
#axiomsClean backwardStores_payload
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
