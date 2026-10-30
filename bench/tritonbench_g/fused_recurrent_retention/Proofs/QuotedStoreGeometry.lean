import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedLoopStores
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Normalization

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Only the existing shape and forward stride facts needed for store layout.
These are derived from the checked call, not new numerical restrictions. -/
structure StoreDomain (s : Shape) (b : InputBuffers) : Prop where
  shape : s.Admitted
  qDense : b.q.strides=contiguous4 s.B s.H s.T s.DK
  vDense : b.v.strides=contiguous4 s.B s.H s.T s.DV

def inputRegion (r : RegionName) : Prop :=
  r="q" ∨ r="k" ∨ r="v" ∨ r="do" ∨ r="initial_state" ∨ r="d_final_state"

def outputRegion (r : RegionName) : Prop :=
  r="o" ∨ r="final_state" ∨ r="dq" ∨ r="dk" ∨ r="dv"

theorem input_output_disjoint (r : RegionName) (hi : inputRegion r) (ho : outputRegion r) : False := by
  rcases hi with rfl|rfl|rfl|rfl|rfl|rfl <;> simp [outputRegion] at ho

theorem covers_iff (items : List (Write α)) (r : RegionName) (i : Nat) :
    covers items r i ↔ ∃ w∈items,r=w.region ∧ i=w.offset := by
  induction items with
  | nil => simp [covers]
  | cons w rest ih => simp [covers,ih]

theorem writes_input_frame (items : List (Write α)) (u : State α)
    (hout : ∀ w∈items,outputRegion w.region) (r : RegionName) (hr : inputRegion r) :
    (writes items u).mem r=u.mem r := by
  funext i
  apply writes_frame
  rw [covers_iff]
  rintro ⟨w,hw,rfl,_⟩
  exact input_output_disjoint w.region hr (hout w hw)

theorem inputs_congr [Inhabited α] (p : Layout) (u v : State (Staged.Node α))
    (h : ∀ r,inputRegion r → u.mem r=v.mem r) : inputs p u=inputs p v := by
  unfold inputs readWord
  congr 1 <;> funext x y z <;> rw [h _ (by simp [inputRegion])]

theorem writes_inputs [Inhabited α] (p : Layout) (items : List (Write (Staged.Node α)))
    (u : State (Staged.Node α)) (hout : ∀ w∈items,outputRegion w.region) :
    inputs p (writes items u)=inputs p u :=
  inputs_congr p _ _ (writes_input_frame items u hout)

theorem filtered_member {ι : Type} (items : List ι) (active : ι → Prop)
    [DecidablePred active] (event : ι → Write α) (w : Write α)
    (hw : w∈items.filterMap (fun i => if active i then some (event i) else none)) :
    ∃ i∈items,active i ∧ w=event i := by
  rcases List.mem_filterMap.mp hw with ⟨i,hi,he⟩
  by_cases ha : active i
  · exact ⟨i,hi,ha,(Option.some.inj (by simpa [ha] using he)).symm⟩
  · simp [ha] at he

theorem outputWrites_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (w : Write (Staged.Node α)) (hw : w∈outputWrites a s b plans initial seed pid t) :
    outputRegion w.region := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,_,rfl⟩
  simp [outputRegion]

theorem queryWrites_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat)
    (w : Write (Staged.Node α)) (hw : w∈queryWrites a s b plans initial seed pid t) :
    outputRegion w.region := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,_,rfl⟩
  simp [outputRegion]

theorem reverseKeyWrites_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (t : Nat)
    (w : Write (Staged.Node α)) (hw : w∈reverseKeyWrites a s b plans initial final pid t) :
    outputRegion w.region := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,_,rfl⟩
  simp [outputRegion]

theorem reverseValueWrites_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (t : Nat)
    (w : Write (Staged.Node α)) (hw : w∈reverseValueWrites a s b plans initial final pid t) :
    outputRegion w.region := by
  rcases filtered_member _ _ _ _ hw with ⟨i,_,_,rfl⟩
  simp [outputRegion]

theorem finalWrites_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (w : Write (Staged.Node α)) (hw : w∈finalWrites a s b initial seed final pid) :
    outputRegion w.region := by
  cases final
  · simp [finalWrites] at hw
  · rcases filtered_member _ _ _ _ hw with ⟨i,_,_,rfl⟩
    simp [outputRegion]

theorem reverseWrites_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (t : Nat)
    (w : Write (Staged.Node α)) (hw : w∈reverseWrites a s b plans initial final pid t) :
    outputRegion w.region := by
  rcases List.mem_append.mp hw with h|h
  · exact reverseKeyWrites_region _ _ _ _ _ _ _ _ w h
  · exact reverseValueWrites_region _ _ _ _ _ _ _ _ w h

theorem forwardStores_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (w : Write (Staged.Node α)) (hw : w∈forwardStores a s b plans initial seed final pid) :
    outputRegion w.region := by
  rcases List.mem_append.mp hw with h|h
  · rcases (mem_timeWrites _ _ _ _).mp h with ⟨t,_,_,h⟩
    exact outputWrites_region _ _ _ _ _ _ _ _ w h
  · exact finalWrites_region _ _ _ _ _ _ _ w h

theorem backwardStores_region [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (w : Write (Staged.Node α)) (hw : w∈backwardStores a s b plans initial seed final pid) :
    outputRegion w.region := by
  rcases List.mem_append.mp hw with h|h
  · rcases (mem_timeWrites _ _ _ _).mp h with ⟨t,_,_,h⟩
    exact queryWrites_region _ _ _ _ _ _ _ _ w h
  · rcases (mem_timeWrites _ _ _ _).mp h with ⟨t,_,_,h⟩
    exact reverseWrites_region _ _ _ _ _ _ _ _ w h

theorem outputOffset_partial (s : Shape) (b : InputBuffers)
    (hd : b.v.strides=contiguous4 s.B s.H s.T s.DV) (ht : 0<s.T) (hv : 0<s.DV)
    (pid : ProgramId) (t lane : Nat) :
    outputOffset (layout s b) pid t lane=partialOffset s s.DV pid.ik pid.head t (pid.value s lane) := by
  unfold outputOffset layout
  simp only [hd]
  rw [source_head_stride s s.DV ht hv]
  unfold partialOffset ProgramId.value Shape.BH
  ring

theorem gradientOffset_partial (s : Shape) (b : InputBuffers)
    (hd : b.q.strides=contiguous4 s.B s.H s.T s.DK) (ht : 0<s.T) (hk : 0<s.DK)
    (pid : ProgramId) (t lane : Nat) :
    gradientOffset (layout s b) pid t lane=partialOffset s s.DK pid.iv pid.head t (pid.key s lane) := by
  unfold gradientOffset layout
  simp only [hd]
  rw [source_head_stride s s.DK ht hk]
  unfold partialOffset ProgramId.key Shape.BH
  ring

theorem seedOffset_state (s : Shape) (b : InputBuffers) (pid : ProgramId) (lk lv : Nat) :
    seedOffset (layout s b) pid lk lv=stateOffset s pid.head (pid.key s lk) (pid.value s lv) := by
  unfold seedOffset layout stateOffset ProgramId.key ProgramId.value
  ring

#axiomsClean writes_inputs
#axiomsClean forwardStores_region
#axiomsClean backwardStores_region
#axiomsClean outputOffset_partial
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
