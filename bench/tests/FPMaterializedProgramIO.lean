import VeriTile.Triton.Float.MaterializedProgramIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.FP.Structural.MaterializedRegression
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 4000

private theorem eval_alg {α : Type} [Inhabited α] (M : Algebra α)
    {dtype : TileDType} {shape : TileShape} (e : Op dtype shape) (s : State α) :
    evalExpr M (.alg e) s = evalOp M none e s := rfl

private theorem safe_alg {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) {dtype : TileDType} {shape : TileShape}
    (e : Op dtype shape) (s : State α) :
    Safety.safeExpr M bounds s (.alg e) = Safety.safeOp M none bounds s e := by
  rw [Safety.safeExpr_unfold] <;> rfl

private theorem trace_singleton {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (stmt : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [stmt] s ↔ Safety.stepSafe M bounds stmt s := by
  rw [Safety.traceSafeList_cons]
  simp

abbrev Pair := Nat × Nat

def producer (M : Algebra Pair) : MaterializedInvocation Nat Pair where
  source := {
    kernel := .mk [] ["middle"] [
      .store .real [] (.region "middle" (.constNat 0)) (.alg (.ref .real [] "value")) .none,
      .store .real [] (.region "middle" (.constNat 1)) (.alg (.ref .real [] "value")) .none]
    backend := M
    arguments := [⟨"value",.real,[],fun _ => (2,7)⟩]
    pids := fun _ => 4
    numPids := some (fun _ => 8) }
  encode := fun x => (x,x)
  decodeAt := fun region offset x => if region = "middle" ∧ offset = 0 then x.1 else x.2
  decode_encode := by intro region offset x; split <;> rfl

def consumer (M : Algebra Pair) : MaterializedInvocation Nat Pair :=
  { producer M with source := {
    kernel := .mk ["middle"] ["out"] [
      .store .real [] (.region "out" (.constNat 0))
        (.alg (.load .real (.region "middle" (.constNat 0)) .none .zero)) .none]
    backend := M
    pids := fun _ => 0 } }

def produced (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) : State Pair :=
  (((producer M).source.initial ((producer M).embedded memory)).write "middle" 0 (.mk .real (2,7))).write
    "middle" 1 (.mk .real (2,7))

def middle (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) : RegionName → Nat → Cell Nat :=
  (producer M).finish (produced M memory).mem

theorem producer_executes (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) :
    (producer M).source.run ((producer M).embedded memory) = some (produced M memory) := by
  simp [producer,produced,Invocation.run,Invocation.initial,bindRegisters,exec,run,step,
    eval_alg,evalOp_unfold,store,TileShape.allIndices,State.setReg,Region.cast]

theorem producer_materializes (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) :
    (producer M).run memory = some (middle M memory) :=
  (producer M).run_eq memory _ (producer_executes M memory)

/-- Two actual stores of the same vector select different scalar owners at
adjacent addresses. Address-independent final observation is insufficient. -/
theorem address_selects_owner (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) :
    middle M memory "middle" 0 = .mk .real 2 ∧
    middle M memory "middle" 1 = .mk .real 7 := by
  simp [middle,MaterializedInvocation.finish,produced,producer,State.write,mapCell,mapValue]

def consumed (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) : State Pair :=
  ((consumer M).source.initial ((consumer M).embedded (middle M memory))).write "out" 0 (.mk .real (2,2))

theorem consumer_executes (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) :
    (consumer M).source.run ((consumer M).embedded (middle M memory)) = some (consumed M memory) := by
  have hmiddle := (address_selects_owner M memory).1
  simp [consumer,producer,consumed,Invocation.run,exec,run,step,eval_alg,evalOp_unfold,
    store,TileShape.allIndices,Region.cast,Invocation.initial,bindRegisters,
    MaterializedInvocation.embedded,MaterializedInvocation.embed,mapCell,mapValue]
  rw [hmiddle]
  rfl

/-- The second CTA sees the first CTA's decoded scalar re-embedded, not its
unmaterialized vector or the schedule-entry memory. -/
theorem successor_consumes_materialized_value (M : Algebra Pair)
    (memory : RegionName → Nat → Cell Nat) :
    ∃ final, runMaterializedSchedule [producer M,consumer M] memory = some final ∧
      final "out" 0 = .mk .real 2 := by
  refine ⟨(consumer M).finish (consumed M memory).mem,?_,?_⟩
  · simp only [runMaterializedSchedule,producer_materializes,Option.bind_some]
    rw [(consumer M).run_eq _ _ (consumer_executes M memory)]
    rfl
  · simp [MaterializedInvocation.finish,consumer,producer,consumed,State.write,mapCell,mapValue]

/-- Delaying materialization to the end would select the wrong replica. -/
theorem delayed_materialization_changes_result (M : Algebra Pair)
    (memory : RegionName → Nat → Cell Nat) :
    ∃ final, runSchedule [(producer M).source,(consumer M).source]
        ((producer M).embedded memory) = some final ∧
      (consumer M).finish final.mem "out" 0 = .mk .real 7 := by
  refine ⟨((consumer M).source.initial (produced M memory)).write "out" 0 (.mk .real (2,7)),?_,?_⟩
  · simp only [runSchedule,producer_executes,Option.bind_some]
    simp [consumer,producer,produced,Invocation.run,exec,run,step,eval_alg,evalOp_unfold,
      store,TileShape.allIndices,Region.cast,Invocation.initial,bindRegisters,State.write]
  · simp [MaterializedInvocation.finish,consumer,producer,State.write,mapCell,mapValue]

/-- Codec round trips keep all discrete payloads and their original tags. -/
theorem discrete_cells_unchanged (M : Algebra Pair) :
    (producer M).finish ((producer M).embed (fun _ _ => .mk .int (-13))) =
      (fun _ _ => .mk .int (-13)) ∧
    (producer M).finish ((producer M).embed (fun _ _ => .mk .ptr ("target",19))) =
      (fun _ _ => .mk .ptr ("target",19)) := by
  constructor <;> exact (producer M).finish_embed _

theorem exact_exterior_cell (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) :
    middle M memory "sentinel" 99 = memory "sentinel" 99 := by
  apply (producer M).finish_frame
  simp [produced,State.write,Invocation.initial,bindRegisters,producer,State.setReg,
    MaterializedInvocation.embedded]

/-- The fixed fresh state is followed by the source's actual ABI setup. -/
theorem fresh_entry_and_abi (M : Algebra Pair) (memory : RegionName → Nat → Cell Nat) :
    ((producer M).source.initial ((producer M).embedded memory)).regs .real [] "value" =
      some (fun _ => (2,7)) ∧
    ((producer M).source.initial ((producer M).embedded memory)).regs .real [] "missing" = none ∧
    ((producer M).source.initial ((producer M).embedded memory)).pids 0 = 4 ∧
    ((producer M).source.initial ((producer M).embedded memory)).numPids 0 = 8 ∧
    ((producer M).source.initial ((producer M).embedded memory)).undef .int "middle" 0 = 0 := by
  simp [producer,Invocation.initial,bindRegisters,State.setReg,MaterializedInvocation.embedded,defaultValue]

def indexBackend (M : Algebra Pair) : Algebra Pair :=
  { M with floatToNat := fun _ _ _ => some (fun x => some x.2) }

def addressConsumer (M : Algebra Pair) : MaterializedInvocation Nat Pair :=
  { producer M with source := {
    kernel := .mk ["middle"] ["out"] [
      .store .nat [] (.region "out" (.floatToNat none .real 32
        (.load .real (.region "middle" (.constNat 0)) .none .zero)))
        (.alg (.constNat 99)) .none]
    backend := indexBackend M
    pids := fun _ => 0 } }

/-- Safety checks the address computed from the actual materialized successor.
The earlier vector's second component is 7; the reached scalar is 2. -/
theorem successor_safety_uses_actual_value (M : Algebra Pair)
    (memory : RegionName → Nat → Cell Nat) (bounds : RegionName → Nat) :
    (addressConsumer M).Safe bounds (middle M memory) ↔
      0 < bounds "middle" ∧ 2 < bounds "out" := by
  have hmiddle := (address_selects_owner M memory).1
  simp [MaterializedInvocation.Safe,Invocation.Safe,TraceSafe,addressConsumer,producer,
    ComputeKernel.surfaceBody,trace_singleton,Safety.stepSafe_unfold,
    Safety.safeAddress,safe_alg,Safety.safeOp_unfold,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,evalOp_unfold,Region.cast,
    Invocation.initial,bindRegisters,MaterializedInvocation.embedded,MaterializedInvocation.embed,
    mapCell,mapValue,indexBackend]
  rw [hmiddle]
  simp [toFloat,mapValue]

def failing (M : Algebra Pair) : MaterializedInvocation Nat Pair :=
  { producer M with source := {
    kernel := .mk [] [] [.assign .real [] "value" (.alg (.ref .real [] "missing"))]
    backend := M
    pids := fun _ => 0 } }

def emptyIO (M : Algebra Pair) : MaterializedProgramIO Nat Nat Pair where
  inputs := []
  outputs := []
  scratch := []
  buffers := []
  read := fun i => Fin.elim0 i
  inputMask := fun i => Fin.elim0 i
  write := fun i => Fin.elim0 i
  outputMask := fun i => Fin.elim0 i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => [failing M]
  encode := id
  observe := fun i => Fin.elim0 i

/-- Empty output lists do not excuse an actual source execution failure. -/
theorem failed_source_rejected (M : Algebra Pair) :
    ¬ (emptyIO M).ImplementsCells (fun _ i => Fin.elim0 i) := by
  intro proof
  let inputs : (emptyIO M).Inputs := fun i => Fin.elim0 i
  let memory : RegionName → Nat → Cell Nat := fun _ _ => .mk .int (-13)
  have bound : (emptyIO M).InputCells inputs memory := by intro i; exact Fin.elim0 i
  obtain ⟨_,final,ran,_⟩ := proof inputs trivial memory bound
  simp [emptyIO,runMaterializedSchedule,MaterializedInvocation.run,MaterializedInvocation.embedded,
    failing,Invocation.run,Invocation.initial,bindRegisters,exec,run,step,eval_alg,evalOp_unfold] at ran

#axiomsClean address_selects_owner
#axiomsClean successor_consumes_materialized_value
#axiomsClean delayed_materialization_changes_result
#axiomsClean discrete_cells_unchanged
#axiomsClean exact_exterior_cell
#axiomsClean fresh_entry_and_abi
#axiomsClean successor_safety_uses_actual_value
#axiomsClean failed_source_rejected
#axiomsClean MaterializedInvocation.finish_embed
#axiomsClean materializedScheduleSafe_append
#axiomsClean MaterializedProgramIO.ImplementsOnCells.intro
end
end VeriTile.Triton.FP.Structural.MaterializedRegression
