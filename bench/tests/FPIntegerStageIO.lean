import VeriTile.Triton.Float.IntegerStageIO

namespace VeriTile.Triton.FP.Structural.IntegerStageRegression
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
open QuantizeIntegerStage

def roundedLoad (active : Bool) : TypedKernel :=
  ⟨.mk ["x"] [] [.assign .int [] "output" (.llrint (.alg
    (.load .real (.region "x" (.constNat 0)) (.mask (.constBool active)) .zero)))],
    fun _ => none, outputRegister⟩

/-- A custom llrint expression must still back its active nested source load. -/
theorem active_llrint_load_rejected {α : Type} [Inhabited α] (M : Algebra α)
    (round : Llrint64 α) (s : State α) :
    ¬ QuantizeIntegerStage.TraceSafe M (some round) (fun _ => 0) (roundedLoad Bool.true) s := by
  simp [QuantizeIntegerStage.TraceSafe, roundedLoad, ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList, QuantizeIntegerStage.Safety.stepSafe,
    QuantizeIntegerStage.Safety.safeExpr, FP.Structural.Safety.safeOp_unfold,
    FP.Structural.Safety.safeAddress, FP.Structural.Safety.safeMask,
    FP.Structural.Safety.addressSafe, FP.Structural.Safety.maskActive, evalOp_unfold]

/-- Its masked-off load needs no backing, exactly like Structural load. -/
theorem inactive_llrint_load_safe {α : Type} [Inhabited α] (M : Algebra α)
    (round : Llrint64 α) (s : State α) :
    QuantizeIntegerStage.TraceSafe M (some round) (fun _ => 0) (roundedLoad Bool.false) s := by
  simp [QuantizeIntegerStage.TraceSafe, roundedLoad, ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList, QuantizeIntegerStage.Safety.stepSafe,
    QuantizeIntegerStage.Safety.safeExpr, FP.Structural.Safety.safeOp_unfold,
    FP.Structural.Safety.safeAddress, FP.Structural.Safety.safeMask,
    FP.Structural.Safety.addressSafe, FP.Structural.Safety.maskActive,
    QuantizeIntegerStage.step, QuantizeIntegerStage.evalExpr, evalOp_unfold]

def indexWrite : ComputeStmt :=
  .store .int [] (.region "index" (.constNat 0)) (.alg (.constInt 257)) .none

def indirectRead : ComputeStmt :=
  .assign .nat [] "read" (.alg (.load .nat
    (.region "data" (.castIntToNat (.load .int (.region "index" (.constNat 0)) .none .zero)))
    .none .zero))

def indirectKernel (narrow : Bool) : TypedKernel :=
  ⟨.mk ["index","data"] ["index"] [indexWrite,indirectRead],
    if narrow then byteOutput "index" else fun _ => none, outputRegister⟩

def bounds (r : RegionName) : Nat := if r="index" then 1 else if r="data" then 2 else 0

/-- The next address reads the narrowed stored byte 1, not the register value 257. -/
theorem narrowed_successor_access_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    QuantizeIntegerStage.TraceSafe M none bounds (indirectKernel Bool.true) s := by
  have narrowed : storedInteger (byteOutput "index") "index" 257 = 1 := rfl
  simp [QuantizeIntegerStage.TraceSafe,indirectKernel,indexWrite,indirectRead,ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList,QuantizeIntegerStage.Safety.stepSafe,
    QuantizeIntegerStage.Safety.safeExpr,FP.Structural.Safety.safeOp_unfold,
    FP.Structural.Safety.safeAddress,FP.Structural.Safety.safeMask,FP.Structural.Safety.addressSafe,
    FP.Structural.Safety.maskActive,QuantizeIntegerStage.step,QuantizeIntegerStage.evalExpr,
    QuantizeIntegerStage.store,FP.Structural.store,evalOp_unfold,TileShape.allIndices,
    State.write,Cell.read,bounds,narrowed,Region.cast]

/-- The same source without INT8 narrowing accesses 257 and is rejected. -/
theorem unnarrowed_successor_access_rejected {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ¬ QuantizeIntegerStage.TraceSafe M none bounds (indirectKernel Bool.false) s := by
  simp [QuantizeIntegerStage.TraceSafe,indirectKernel,indexWrite,indirectRead,ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList,QuantizeIntegerStage.Safety.stepSafe,
    QuantizeIntegerStage.Safety.safeExpr,FP.Structural.Safety.safeOp_unfold,
    FP.Structural.Safety.safeAddress,FP.Structural.Safety.safeMask,FP.Structural.Safety.addressSafe,
    FP.Structural.Safety.maskActive,QuantizeIntegerStage.step,QuantizeIntegerStage.evalExpr,
    QuantizeIntegerStage.store,FP.Structural.store,evalOp_unfold,TileShape.allIndices,
    State.write,Cell.read,bounds,storedInteger,Region.cast]

def failingJob (M : Algebra Nat) : IntegerInvocation Nat where
  kernel := .mk [] [] [.assign .int [] "output" (.llrint (.alg (.ref .real [] "x")))]
  backend := M
  integerStorage := fun _ => none
  runtimeRegisters := outputRegister
  round := none
  pids := fun _ => 0

def failingIO (M : Algebra Nat) : IntegerProgramIO Nat Nat where
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
  schedule := fun _ => [failingJob M]
  encode := id
  observe := fun i => Fin.elim0 i

def sameVariant (M : Algebra Nat) : IntegerProgramIO.Variant (failingIO M) where
  schedule := (failingIO M).schedule
  observe := (failingIO M).observe

/-- Missing llrint cannot be disguised as equivalent execution of two failures. -/
theorem missing_primitive_equivalence_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (failingIO M).Equivalent (sameVariant M) := by
  intro h
  let vals : (failingIO M).Inputs := fun i => Fin.elim0 i
  have hb : (failingIO M).InBounds (fun _ => 0) vals := by
    refine ⟨fun _ _ => rfl, ?_, ?_, ?_⟩ <;> intro i <;> exact Fin.elim0 i
  have hi : (failingIO M).InputBinding vals s := by intro i; exact Fin.elim0 i
  obtain ⟨_, _, left, right, executed, _⟩ := h vals trivial (fun _ => 0) hb s hi
  simp [failingIO,failingJob,runIntegerSchedule,IntegerInvocation.run,IntegerInvocation.typedKernel,
    QuantizeIntegerStage.exec,QuantizeIntegerStage.run,QuantizeIntegerStage.step,
    QuantizeIntegerStage.evalExpr,ComputeKernel.surfaceBody] at executed

def dependentJobs {α : Type} (M : Algebra α) : List (IntegerInvocation α) :=
  [{ kernel := .mk [] ["index"] [indexWrite], backend := M,
     integerStorage := byteOutput "index", runtimeRegisters := outputRegister,
     round := none, pids := fun _ => 0 },
   { kernel := .mk ["index","data"] [] [indirectRead], backend := M,
     integerStorage := fun _ => none, runtimeRegisters := outputRegister,
     round := none, pids := fun _ => 0 }]

/-- A later launch also reads the actual narrowed predecessor memory. -/
theorem schedule_uses_predecessor_memory {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) : IntegerScheduleSafe bounds (dependentJobs M) s := by
  have narrowed : storedInteger (byteOutput "index") "index" 257 = 1 := rfl
  simp [IntegerScheduleSafe,dependentJobs,IntegerInvocation.Safe,IntegerInvocation.run,
    IntegerInvocation.typedKernel,IntegerInvocation.initial,Invocation.initial,bindRegisters,
    QuantizeIntegerStage.TraceSafe,QuantizeIntegerStage.exec,QuantizeIntegerStage.run,
    indexWrite,indirectRead,ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList,QuantizeIntegerStage.Safety.stepSafe,
    QuantizeIntegerStage.Safety.safeExpr,FP.Structural.Safety.safeOp_unfold,
    FP.Structural.Safety.safeAddress,FP.Structural.Safety.safeMask,FP.Structural.Safety.addressSafe,
    FP.Structural.Safety.maskActive,QuantizeIntegerStage.step,QuantizeIntegerStage.evalExpr,
    QuantizeIntegerStage.store,FP.Structural.store,evalOp_unfold,TileShape.allIndices,
    State.write,Cell.read,bounds,narrowed,Region.cast]

end
end VeriTile.Triton.FP.Structural.IntegerStageRegression
