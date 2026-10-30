import bench.repairs.tritonbench_g.chunk_linear_attn.Readback
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def readRegions : Stage → List RegionName
  | .forwardState | .backwardState => ["query","key","value","incoming","initial","terminalGradient"]
  | .forwardOutput => ["query","key","value","history"]
  | .backwardGradient => ["query","key","value","incoming","history","historyGradient"]
def SameReads (stage : Stage) (a b : Memory α) := ∀ r, r∈readRegions stage → ∀ o, a r o=b r o

theorem SameReads.refl (stage : Stage) (m : Memory α) : SameReads stage m m := by intro r hr o; rfl
theorem SameReads.trans {stage : Stage} {a b c : Memory α} (ab : SameReads stage a b) (bc : SameReads stage b c) : SameReads stage a c := by
  intro r hr o; exact (ab r hr o).trans (bc r hr o)

theorem reads_not_written (stage : Stage) (p : Parameters) (final : Bool) (pid : PID) (r : RegionName) (o : Nat)
    (hr : r∈readRegions stage) : ¬Footprint stage p final pid r o := by
  intro hw
  cases stage with
  | forwardState =>
    rcases hw with ⟨rfl,_⟩|⟨rfl,_⟩ <;> simp [readRegions,historyName,finalName] at hr
  | backwardState =>
    rcases hw with ⟨rfl,_⟩|⟨rfl,_⟩ <;> simp [readRegions,historyName,finalName] at hr
  | forwardOutput =>
    rcases hw with ⟨rfl,_⟩
    simp [readRegions] at hr
  | backwardGradient =>
    rcases hw with ⟨hr',_⟩|⟨rfl,_⟩
    · rcases hr' with rfl|rfl <;> simp [readRegions] at hr
    · simp [readRegions] at hr

theorem stage_sameReads (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (m : Memory α) :
    SameReads stage (stageMemory M D stage p scale seed final pid m) m := by
  intro r hr o
  exact stage_frame M D stage p scale seed final pid m r o (reads_not_written stage p final pid r o hr)

theorem grid_sameReads (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (schedule : List PID) (m : Memory α) :
    SameReads stage (gridMemory M D stage p scale seed final schedule m) m := by
  intro r hr o
  apply grid_frame
  intro pid _
  exact reads_not_written stage p final pid r o hr

theorem stateNext_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (rev : Bool) (n : Nat) (v : HValues p α) (h : StateInputs a b) :
    stateNext M D p pid a scale rev n v=stateNext M D p pid b scale rev n v := by
  have hq := h "query" (by simp); have hk := h "key" (by simp)
  have hv := h "value" (by simp); have hg := h "incoming" (by simp)
  unfold stateNext stateLeft stateRight load rawLoad
  simp only [hq,hk,hv,hg]

theorem stateValues_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (rev seed : Bool) (h : StateInputs a b) (n : Nat) :
    stateValues M D p pid a scale rev seed n=stateValues M D p pid b scale rev seed n := by
  induction n with
  | zero => exact stateInitial_inputs M p pid a b rev seed h
  | succ n ih => simp only [stateValues,ih,stateNext_inputs M D p pid a b scale rev n _ h]

theorem outputNext_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (kb : Nat) (v : OutputAcc p α) (h : SameReads .forwardOutput a b) :
    outputNext M D p pid a kb v=outputNext M D p pid b kb v := by
  have hq := h "query" (by simp [readRegions]); have hk := h "key" (by simp [readRegions])
  have hh := h "history" (by simp [readRegions])
  unfold outputNext queryTile keyTile forwardHistoryTile load rawLoad
  simp only [hq,hk,hh]

theorem outputStates_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (h : SameReads .forwardOutput a b) (n : Nat) :
    outputStates M D p pid a n=outputStates M D p pid b n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [outputStates,ih,outputNext_inputs M D p pid a b n _ h]

theorem outputResult_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (v : OutputAcc p α) (h : SameReads .forwardOutput a b) :
    outputResult M D p pid a scale v=outputResult M D p pid b scale v := by
  have hv := h "value" (by simp [readRegions])
  unfold outputResult valueTile load rawLoad
  simp only [hv]

theorem gradNext_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (vb : Nat) (v : GradAcc p α) (h : GradientInputs a b) :
    gradNext M D p pid a scale vb v=gradNext M D p pid b scale vb v := by
  have hv := h "value" (by simp); have hg := h "incoming" (by simp)
  have hh := h "history" (by simp); have hd := h "historyGradient" (by simp)
  unfold gradNext valueTile gradHistory gradHistoryGradient load rawLoad
  simp only [hv,hg,hh,hd]

theorem gradStates_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (h : GradientInputs a b) (n : Nat) :
    gradStates M D p pid a scale n=gradStates M D p pid b scale n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [gradStates,ih,gradNext_inputs M D p pid a b scale n _ h]

theorem gradValue_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (vb : Nat) (h : GradientInputs a b) :
    gradValue M D p pid a scale vb=gradValue M D p pid b scale vb := by
  have hq := h "query" (by simp); have hk := h "key" (by simp)
  have hg := h "incoming" (by simp); have hd := h "historyGradient" (by simp)
  unfold gradValue gradScore gradKey gradQuery gradHistoryGradient valueTile load rawLoad
  simp only [hq,hk,hg,hd]

theorem gradResults_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (scale : ℝ) (v : GradAcc p α) (h : GradientInputs a b) :
    gradQueryResult M D p pid a scale v=gradQueryResult M D p pid b scale v ∧
    gradKeyResult M D p pid a scale v=gradKeyResult M D p pid b scale v := by
  have hq := h "query" (by simp); have hk := h "key" (by simp)
  unfold gradQueryResult gradKeyResult gradKey gradQuery load rawLoad
  simp only [hq,hk,and_self]

#axiomsClean stage_sameReads
#axiomsClean grid_sameReads
#axiomsClean stateValues_inputs
#axiomsClean outputStates_inputs
#axiomsClean gradStates_inputs
#axiomsClean gradResults_inputs
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
