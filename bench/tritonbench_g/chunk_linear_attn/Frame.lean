import bench.tritonbench_g.chunk_linear_attn.Grid
import bench.tritonbench_g.chunk_delta_fwd.Sources.StoreRange
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe foldRange_last scatter_readback)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def StateFootprint (p : Parameters) (pid : PID) (rev final : Bool) (r : RegionName) (o : Nat) : Prop :=
  (r=historyName rev ∧ ∃ c, c<p.NT ∧ ∃ i : TileIndex [p.BK,p.BV], hActive p pid i ∧ o=hOffset p pid c i) ∨
  (r=finalName rev ∧ final=Bool.true ∧ ∃ i : TileIndex [p.BK,p.BV], hActive p pid i ∧ o=fOffset p pid i)
def OutputFootprint (p : Parameters) (pid : PID) (r : RegionName) (o : Nat) : Prop :=
  r="output" ∧ ∃ i : TileIndex [p.BT,p.BV], outputActive p pid i ∧ o=outputOffset p pid i
def GradientFootprint (p : Parameters) (pid : PID) (r : RegionName) (o : Nat) : Prop :=
  ((r="queryGradient" ∨ r="keyGradient") ∧ ∃ i : TileIndex [p.BT,p.BK], gradKeyActive p pid i ∧ o=gradKeyOffset p pid i) ∨
  (r="valuePartials" ∧ ∃ vb, vb<valueBlocks p ∧ ∃ i : TileIndex [p.BT,p.BV],
    gradValueActive p pid vb i ∧ o=gradValueOffset p pid (p.B*p.H) vb i)
def Footprint (stage : Stage) (p : Parameters) (final : Bool) (pid : PID) (r : RegionName) (o : Nat) : Prop :=
  match stage with
  | .forwardState => StateFootprint p pid Bool.false final r o
  | .backwardState => StateFootprint p pid Bool.true final r o
  | .forwardOutput => OutputFootprint p pid r o
  | .backwardGradient => GradientFootprint p pid r o

theorem state_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (rev seed final : Bool) (r : RegionName) (o : Nat)
    (safe : ¬StateFootprint p pid rev final r o) :
    stateMemory M D p pid input m scale rev seed final r o=m r o := by
  unfold stateMemory
  have hf : ∀ mem, writeStateFinal M p pid rev final (stateValues M D p pid input scale rev seed p.NT) mem r o=mem r o := by
    intro mem
    cases hf : final
    · simp [writeStateFinal,hf]
    · apply scatter_frame
      by_cases hr : r=finalName rev
      · right
        intro i _ hi he
        exact safe (Or.inr ⟨hr,hf,i,hi,he⟩)
      · exact Or.inl hr
  rw [hf]
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro n hn mem
  apply scatter_frame
  by_cases hr : r=historyName rev
  · right
    intro i _ hi he
    exact safe (Or.inl ⟨hr,chunk p rev n,Geometry.chunk_lt p rev n hn,i,hi,he⟩)
  · exact Or.inl hr

theorem output_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (r : RegionName) (o : Nat)
    (safe : ¬OutputFootprint p pid r o) : outputMemory M D p pid input m scale r o=m r o := by
  apply scatter_frame
  by_cases hr : r="output"
  · right
    intro i _ hi he
    exact safe ⟨hr,i,hi,he⟩
  · exact Or.inl hr

theorem gradient_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (r : RegionName) (o : Nat)
    (safe : ¬GradientFootprint p pid r o) : gradientMemory M D p pid (p.B*p.H) input m scale r o=m r o := by
  have hk : ∀ (dest : RegionName), (dest="queryGradient" ∨ dest="keyGradient") → ∀ (a : Matrix α p.BT p.BK) mem,
      gradKeyWrite M p pid dest a mem r o=mem r o := by
    intro dest hd a mem
    apply scatter_frame
    by_cases hr : r=dest
    · right
      intro i _ hi he
      exact safe (Or.inl ⟨hr ▸ hd,i,hi,he⟩)
    · exact Or.inl hr
  unfold gradientMemory
  rw [hk "keyGradient" (Or.inr rfl),hk "queryGradient" (Or.inl rfl)]
  apply foldRange_observe _ (fun mem : Memory α => mem r o)
  intro vb hv mem
  apply scatter_frame
  by_cases hr : r="valuePartials"
  · right
    intro i _ hi he
    exact safe (Or.inr ⟨hr,vb,hv,i,hi,he⟩)
  · exact Or.inl hr

theorem stage_frame (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (m : Memory α) (r : RegionName) (o : Nat)
    (safe : ¬Footprint stage p final pid r o) : stageMemory M D stage p scale seed final pid m r o=m r o := by
  cases stage
  · exact state_frame M D p pid m m scale Bool.false seed final r o safe
  · exact output_frame M D p pid m m scale r o safe
  · exact state_frame M D p pid m m scale Bool.true seed final r o safe
  · exact gradient_frame M D p pid m m scale r o safe

theorem list_observe {β γ ι : Type} (f : β → ι → β) (get : β → γ)
    (xs : List ι) (a : β) (h : ∀ i∈xs, ∀ b, get (f b i)=get b) : get (xs.foldl f a)=get a := by
  induction xs generalizing a with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih]
    · exact h i (by simp) a
    · intro j hj
      exact h j (by simp [hj])

theorem grid_frame (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory α) (r : RegionName) (o : Nat)
    (safe : ∀ pid∈schedule, ¬Footprint stage p final pid r o) : gridMemory M D stage p scale seed final schedule m r o=m r o := by
  apply list_observe _ (fun mem : Memory α => mem r o) schedule m
  intro pid hp mem
  exact stage_frame M D stage p scale seed final pid mem r o (safe pid hp)

#axiomsClean state_frame
#axiomsClean output_frame
#axiomsClean gradient_frame
#axiomsClean grid_frame
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
