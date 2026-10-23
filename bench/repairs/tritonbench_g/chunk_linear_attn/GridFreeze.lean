import bench.repairs.tritonbench_g.chunk_linear_attn.InputPreservation
import bench.repairs.tritonbench_g.chunk_delta_fwd.Grid
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

/-- A proof-only snapshot: equality with the actual current-memory execution is
derived below from the input-region frame, never postulated. -/
def frozenStage (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (input current : Memory α) : Memory α :=
  match stage with
  | .forwardState => stateMemory M D p pid input current scale Bool.false seed final
  | .forwardOutput => outputMemory M D p pid input current scale
  | .backwardState => stateMemory M D p pid input current scale Bool.true seed final
  | .backwardGradient => gradientMemory M D p pid (p.B*p.H) input current scale

theorem stage_freeze (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (input current : Memory α)
    (h : SameReads stage current input) :
    stageMemory M D stage p scale seed final pid current=
      frozenStage M D stage p scale seed final pid input current := by
  cases stage with
  | forwardState =>
    unfold stageMemory frozenStage stateMemory
    simp only [stateValues_inputs M D p pid current input scale Bool.false seed h]
  | backwardState =>
    unfold stageMemory frozenStage stateMemory
    simp only [stateValues_inputs M D p pid current input scale Bool.true seed h]
  | forwardOutput =>
    unfold stageMemory frozenStage outputMemory
    rw [outputStates_inputs M D p pid current input h]
    rw [outputResult_inputs M D p pid current input scale _ h]
  | backwardGradient =>
    unfold stageMemory frozenStage gradientMemory
    simp only [gradStates_inputs M D p pid current input scale h,
      (gradResults_inputs M D p pid current input scale _ h).1,
      (gradResults_inputs M D p pid current input scale _ h).2,
      gradValue_inputs M D p pid current input scale _ h]

def frozenGrid (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (input current : Memory α) : Memory α :=
  schedule.foldl (fun mem pid => frozenStage M D stage p scale seed final pid input mem) current

theorem grid_freeze (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (input current : Memory α)
    (h : SameReads stage current input) :
    gridMemory M D stage p scale seed final schedule current=
      frozenGrid M D stage p scale seed final schedule input current := by
  induction schedule generalizing current with
  | nil => rfl
  | cons pid rest ih =>
    change gridMemory M D stage p scale seed final rest
        (stageMemory M D stage p scale seed final pid current)=_
    rw [ih _ ((stage_sameReads M D stage p scale seed final pid current).trans h)]
    rw [stage_freeze M D stage p scale seed final pid input current h]
    rfl

theorem frozen_frame (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (input current : Memory α) (r : RegionName) (o : Nat)
    (h : ¬Footprint stage p final pid r o) :
    frozenStage M D stage p scale seed final pid input current r o=current r o := by
  cases stage with
  | forwardState => exact state_frame M D p pid input current scale Bool.false seed final r o h
  | backwardState => exact state_frame M D p pid input current scale Bool.true seed final r o h
  | forwardOutput => exact output_frame M D p pid input current scale r o h
  | backwardGradient => exact gradient_frame M D p pid input current scale r o h

theorem state_footprint_owner (p : Parameters) (rev final : Bool) (a b : PID)
    (r : RegionName) (o : Nat) (ha : StateFootprint p a rev final r o)
    (hb : StateFootprint p b rev final r o) : a=b := by
  rcases ha with ⟨ra,c,hc,i,hi,he⟩|⟨ra,hf,i,hi,he⟩
  · rcases hb with ⟨rb,d,hd,j,hj,he'⟩|⟨rb,_,j,hj,he'⟩
    · exact (Geometry.history_collision p a b c d i j hi hj hc hd (he.symm.trans he')).1
    · have h := ra.symm.trans rb
      cases rev <;> simp [historyName,finalName] at h
  · rcases hb with ⟨rb,d,hd,j,hj,he'⟩|⟨rb,_,j,hj,he'⟩
    · have h := ra.symm.trans rb
      cases rev <;> simp [historyName,finalName] at h
    · exact (Geometry.final_collision p a b i j hi hj (he.symm.trans he')).1

theorem footprint_owner (stage : Stage) (p : Parameters) (final : Bool) (a b : PID)
    (ha : ValidPID stage p a) (hb : ValidPID stage p b)
    (r : RegionName) (o : Nat) (wa : Footprint stage p final a r o)
    (wb : Footprint stage p final b r o) : a=b := by
  cases stage with
  | forwardState => exact state_footprint_owner p Bool.false final a b r o wa wb
  | backwardState => exact state_footprint_owner p Bool.true final a b r o wa wb
  | forwardOutput =>
    rcases wa with ⟨_,i,hi,he⟩
    rcases wb with ⟨_,j,hj,he'⟩
    exact (Geometry.output_collision p a b i j hi hj (he.symm.trans he')).1
  | backwardGradient =>
    rcases wa with ⟨ra,i,hi,he⟩|⟨ra,v,hv,i,hi,he⟩
    · rcases wb with ⟨_,j,hj,he'⟩|⟨rb,v,hv,j,hj,he'⟩
      · exact (Geometry.gradient_collision p a b i j hi hj (he.symm.trans he')).1
      · rcases ra with rfl|rfl <;> simp at rb
    · rcases wb with ⟨rb,j,hj,he'⟩|⟨_,v',hv',j,hj,he'⟩
      · rcases rb with rfl|rfl <;> simp at ra
      · exact (Geometry.partial_collision p (p.B*p.H) a b v v' i j ha.2.2 hb.2.2 hi hj (he.symm.trans he')).1

theorem grid_read (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (input : Memory α)
    (pid : PID) (r : RegionName) (o : Nat) (value : Cell α)
    (valid : ∀ x∈schedule, ValidPID stage p x) (hm : pid∈schedule)
    (hw : Footprint stage p final pid r o)
    (hit : ∀ mem, frozenStage M D stage p scale seed final pid input mem r o=value) :
    gridMemory M D stage p scale seed final schedule input r o=value := by
  classical
  rw [grid_freeze M D stage p scale seed final schedule input input (SameReads.refl _ _)]
  apply VeriTile.Bench.Repairs.ChunkDeltaFwd.list_read
    _ (fun mem : Memory α => mem r o) schedule pid input value hm hit
  intro other ho hn mem
  apply frozen_frame
  intro hw'
  exact hn (footprint_owner stage p final other pid (valid other ho) (valid pid hm) r o hw' hw)

#axiomsClean stage_freeze
#axiomsClean grid_freeze
#axiomsClean footprint_owner
#axiomsClean grid_read
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
