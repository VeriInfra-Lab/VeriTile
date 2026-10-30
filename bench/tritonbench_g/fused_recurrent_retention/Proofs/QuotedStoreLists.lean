import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedWrites

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

def outputWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) : List (Write (Staged.Node α)) :=
  (TileShape.allIndices [s.BV]).filterMap (fun i => if pid.value s i.1.val<s.DV then
    some ⟨"o",outputOffset (layout s b) pid t i.1.val,
      .mk .real (Staged.atom (programO a s plans (inputs (layout s b) initial) seed pid t i.1.val))⟩ else none)

def finalWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) : List (Write (Staged.Node α)) :=
  if final then (TileShape.allIndices [s.BV,s.BK]).filterMap (fun i =>
    if pid.key s i.2.1.val<s.DK ∧ pid.value s i.1.val<s.DV then
      some ⟨"final_state",seedOffset (layout s b) pid i.2.1.val i.1.val,
        .mk .real (Staged.atom (programForwardCarry a s (inputs (layout s b) initial) seed pid s.T i.2.1.val i.1.val))⟩
    else none) else []

def queryWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) : List (Write (Staged.Node α)) :=
  (TileShape.allIndices [s.BK]).filterMap (fun i => if pid.key s i.1.val<s.DK then
    some ⟨"dq",gradientOffset (layout s b) pid t i.1.val,
      .mk .real (Staged.atom (programQ a s plans (inputs (layout s b) initial) seed pid t i.1.val))⟩ else none)

def reverseKeyWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) : List (Write (Staged.Node α)) :=
  (TileShape.allIndices [s.BK]).filterMap (fun i => if pid.key s i.1.val<s.DK then
    some ⟨"dk",reverseGradientOffset (layout s b) pid r i.1.val,
      .mk .real (Staged.atom (programK a s plans (inputs (layout s b) initial) final pid (s.T-1-r) i.1.val))⟩ else none)

def reverseValueWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) : List (Write (Staged.Node α)) :=
  (TileShape.allIndices [s.BV]).filterMap (fun i => if pid.value s i.1.val<s.DV then
    some ⟨"dv",reverseOutputOffset (layout s b) pid r i.1.val,
      .mk .real (Staged.atom (programV a s plans (inputs (layout s b) initial) final pid (s.T-1-r) i.1.val))⟩ else none)

def reverseWrites [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) : List (Write (Staged.Node α)) :=
  reverseKeyWrites a s b plans initial final pid r++reverseValueWrites a s b plans initial final pid r

theorem outputWrites_state [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    writes (outputWrites a s b plans initial seed pid t) u=forwardScatter a s b plans initial seed pid t u := by
  exact scatter_writes _ _ _ _

theorem queryWrites_state [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    writes (queryWrites a s b plans initial seed pid t) u=backwardFirstScatter a s b plans initial seed pid t u := by
  exact scatter_writes _ _ _ _

theorem reverseWrites_state [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial u : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) :
    writes (reverseWrites a s b plans initial final pid r) u=reverseScatter a s b plans initial final pid r u := by
  unfold reverseWrites
  rw [writes_append]
  unfold reverseKeyWrites reverseValueWrites reverseScatter
  rw [scatter_writes,scatter_writes]

theorem forwardBodyResult_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    (forwardBodyResult a s b plans initial u seed pid t).mem=
      (writes (outputWrites a s b plans initial seed pid t) u).mem := by
  unfold forwardBodyResult
  simp only [State.setReg_mem]
  rw [←outputWrites_state]
  apply writes_mem_congr
  simp [forwardBeforeStore]

theorem backwardFirstBodyResult_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial u : State (Staged.Node α)) (seed : Bool) (pid : ProgramId) (t : Nat) :
    (backwardFirstBodyResult a s b plans initial u seed pid t).mem=
      (writes (queryWrites a s b plans initial seed pid t) u).mem := by
  unfold backwardFirstBodyResult
  simp only [State.setReg_mem]
  rw [←queryWrites_state]
  apply writes_mem_congr
  simp [backwardFirstBeforeStore]

theorem reverseBodyResult_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (initial u : State (Staged.Node α)) (final : Bool) (pid : ProgramId) (r : Nat) :
    (reverseBodyResult a s b plans initial u final pid r).mem=
      (writes (reverseWrites a s b plans initial final pid r) u).mem := by
  unfold reverseBodyResult
  simp only [State.setReg_mem]
  rw [←reverseWrites_state]
  apply writes_mem_congr
  simp [reverseBeforeStores]

theorem forwardFinalResult_mem [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial u : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) :
    (forwardFinalResult a s b initial u seed final pid).mem=
      (writes (finalWrites a s b initial seed final pid) u).mem := by
  cases final
  · rfl
  · unfold forwardFinalResult finalWrites
    simp only [Bool.true_eq,↓reduceIte]
    rw [←scatter_writes (active := fun (i : TileIndex [s.BV,s.BK]) =>
      pid.key s i.2.1.val<s.DK ∧ pid.value s i.1.val<s.DV) (event := fun (i : TileIndex [s.BV,s.BK]) =>
      ⟨"final_state",seedOffset (layout s b) pid i.2.1.val i.1.val,
        .mk .real (Staged.atom (programForwardCarry a s (inputs (layout s b) initial) seed pid s.T i.2.1.val i.1.val))⟩)]
    apply writes_mem_congr
    rfl

#axiomsClean outputWrites_state
#axiomsClean reverseWrites_state
#axiomsClean forwardBodyResult_mem
#axiomsClean reverseBodyResult_mem
#axiomsClean forwardFinalResult_mem
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
