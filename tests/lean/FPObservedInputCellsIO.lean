import VeriTile.Triton.Float.ObservedInputCellsIO
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Triton.FP.Structural.ObservedInputCellsRegression
open Safety
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
abbrev Symbol := Nat × Nat

inductive Kind where
  | good | failure | oob | wrongDtype | exterior
  deriving DecidableEq

def kernel : Kind → ComputeKernel
  | .failure => .mk ["x"] ["out"] [.assign .nat [] "v" (.alg (.ref .nat [] "missing"))]
  | .oob => .mk ["x"] ["out"] [.assign .fp32 [] "v"
      (.alg (.load .fp32 (.region "x" (.constNat 1)) .none .zero))]
  | .wrongDtype => .mk ["x"] ["out"] [.store .real [] (.region "out" (.constNat 0))
      (.alg (.ref .real [] "realValue")) .none]
  | .exterior => .mk ["x"] ["out"] [.store .fp32 [] (.region "secret" (.constNat 0))
      (.alg (.ref .fp32 [] "value")) .none]
  | .good => .mk ["x"] ["out"] [.store .fp32 [] (.region "out" (.constNat 0))
      (.alg (.ref .fp32 [] "value")) .none]

def job (M : Algebra Symbol) (kind : Kind) : Invocation Symbol where
  kernel := kernel kind
  backend := M
  clearRegisters := true
  arguments := [⟨"value",.fp32,[],fun _ => (7,0)⟩,⟨"realValue",.real,[],fun _ => (7,0)⟩]
  pids := fun _ => 0

def io (M : Algebra Symbol) (kind : Kind) : ProgramIO Symbol Symbol where
  inputs := [⟨.fp32,[],"x"⟩]
  outputs := [⟨.fp32,[],"out"⟩]
  buffers := ["x","out","secret"]
  read := fun _ _ _ => 0
  write := fun _ _ _ => 0
  inputMask := fun _ _ _ => True
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [job M kind]
  encode := id
  observe := fun _ _ => id

def values (M : Algebra Symbol) (kind : Kind) : (io M kind).Inputs
  | ⟨0,_⟩ => fun _ => (7,42)
  | ⟨i+1,h⟩ => False.elim (by simp [io] at h)
def expected (M : Algebra Symbol) (kind : Kind) : (io M kind).Outputs
  | ⟨0,_⟩ => fun _ => (7,0)
  | ⟨i+1,h⟩ => False.elim (by simp [io] at h)
def constant (M : Algebra Symbol) (kind : Kind) :
    (o : Fin (io M kind).outputs.length) → TileIndex (io M kind).outputs[o].shape → Symbol → Nat :=
  fun _ _ _ => 0
def result (M : Algebra Symbol) (kind : Kind) :
    (io M kind).Inputs → (io M kind).ObservedOutputs Nat :=
  fun _ => (io M kind).mapOutputs (constant M kind) (expected M kind)

def state : State Symbol where
  mem := fun _ _ => .mk .fp32 (7,42)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d
def bounds : RegionName → Nat := fun r => if r="x" ∨ r="out" ∨ r="secret" then 1 else 0

theorem witness (M : Algebra Symbol) (kind : Kind) :
    (io M kind).InBounds bounds (values M kind) ∧
    (io M kind).InputCells (values M kind) state := by
  constructor
  · refine ⟨?_,?_,?_,?_⟩
    · intro r outside
      simpa [io] using (show bounds r=0 from by
        simp [bounds,show ¬(r="x" ∨ r="out" ∨ r="secret") from by simpa [io] using outside])
    · intro i j _; fin_cases i; simp [io,bounds]
    · intro o j _; fin_cases o; simp [io,bounds]
    · intro k; exact Fin.elim0 k
  · intro i j _; fin_cases i; rfl

theorem failed_source_rejected (M : Algebra Symbol) :
    ¬ (io M .failure).ObservesCells (constant M .failure) (result M .failure) := by
  intro implemented
  obtain ⟨_,final,ran,_⟩ := implemented (values M .failure) trivial bounds
    (witness M .failure).1 state (witness M .failure).2
  simp [io,job,kernel,runSchedule,Invocation.run,Invocation.initial,bindRegisters,
    exec,run,step,evalExpr,evalOp_unfold,ComputeKernel.surfaceBody] at ran

theorem oob_source_rejected (M : Algebra Symbol) :
    ¬ (io M .oob).ObservesCells (constant M .oob) (result M .oob) := by
  intro implemented
  have safe := (implemented (values M .oob) trivial bounds
    (witness M .oob).1 state (witness M .oob).2).1.1
  simp [Invocation.Safe,TraceSafe,io,job,kernel,Invocation.initial,bindRegisters,
    ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    evalOp_unfold,Region.cast,bounds] at safe

theorem wrong_dtype_rejected (M : Algebra Symbol) :
    ¬ (io M .wrongDtype).ObservesCells (constant M .wrongDtype) (result M .wrongDtype) := by
  intro implemented
  obtain ⟨_,final,ran,output,_⟩ := implemented (values M .wrongDtype) trivial bounds
    (witness M .wrongDtype).1 state (witness M .wrongDtype).2
  simp [io,job,kernel,runSchedule,Invocation.run,Invocation.initial,bindRegisters,
    exec,run,step,evalExpr,evalOp_unfold,ComputeKernel.surfaceBody,store,evalOffsets,Option.bind] at ran
  cases ran
  have wrong := output (0 : Fin 1) PUnit.unit trivial
  simp [ProgramIO.ObservedOutputBinding,ProgramIO.mapOutputs,io,constant,result,expected,
    mapCell,mapValue,State.write] at wrong

theorem raw_frame_not_hidden (M : Algebra Symbol) :
    ¬ (io M .exterior).ObservesCells (constant M .exterior) (result M .exterior) := by
  intro implemented
  obtain ⟨_,final,ran,_,frame⟩ := implemented (values M .exterior) trivial bounds
    (witness M .exterior).1 state (witness M .exterior).2
  have outside : (io M .exterior).Outside (values M .exterior) "secret" 0 := by
    constructor
    · intro o j _; fin_cases o; left; change ("secret" : RegionName)≠"out"; decide
    · intro k; exact Fin.elim0 k
  have preserved := frame "secret" 0 outside
  simp [io,job,kernel,runSchedule,Invocation.run,Invocation.initial,bindRegisters,
    exec,run,step,evalExpr,evalOp_unfold,ComputeKernel.surfaceBody,store,evalOffsets,Option.bind] at ran
  cases ran
  simp [State.write,state] at preserved

theorem exact_input_dtype_not_relaxed (M : Algebra Symbol) :
    ¬ (io M .good).InputCells (values M .good) {state with mem:=fun _ _=>.mk .real (7,42)} := by
  intro input
  have h := input (0 : Fin 1) PUnit.unit trivial
  cases h

theorem good_implementation (M : Algebra Symbol) :
    (io M .good).ImplementsCells (fun _=>expected M .good) := by
  intro vals _ capacity backed initial _
  have outBound := backed.2.2.1 (0 : Fin 1) PUnit.unit trivial
  let final := ((job M .good).initial initial).write "out" 0 (.mk .fp32 (7,0))
  refine ⟨?_,final,?_,?_,?_⟩
  · change (job M .good).Safe capacity initial ∧ _
    refine ⟨?_,fun _ _ => trivial⟩
    simp [Invocation.Safe,TraceSafe,job,kernel,ComputeKernel.surfaceBody,Invocation.initial,
      bindRegisters,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      evalOp_unfold,Region.cast,step,evalExpr,store,evalOffsets,Option.bind]
    exact outBound
  · simp [io,job,kernel,runSchedule,Invocation.run,Invocation.initial,bindRegisters,
      exec,run,step,evalExpr,evalOp_unfold,ComputeKernel.surfaceBody,store,evalOffsets,Option.bind,final]
  · intro o j _
    fin_cases o
    simp [final,State.write,ProgramIO.OutputBinding,io,expected,mapCell,mapValue]
    rfl
  · intro r a outside
    have h := outside.1 (0 : Fin 1) PUnit.unit trivial
    change r≠"out" ∨ a≠0 at h
    rcases h with hr|ha
    · simp [final,State.write,hr]
    · simp [final,State.write,ha]

/-- An actual successful source store composes with a non-injective observer. -/
theorem composition (M : Algebra Symbol) :
    (io M .good).ObservesCells (constant M .good) (result M .good) :=
  ProgramIO.ObservesCells.of_implementsCells (good_implementation M) (constant M .good)

#axiomsClean witness
#axiomsClean failed_source_rejected
#axiomsClean oob_source_rejected
#axiomsClean wrong_dtype_rejected
#axiomsClean raw_frame_not_hidden
#axiomsClean exact_input_dtype_not_relaxed
#axiomsClean good_implementation
#axiomsClean composition
end
end VeriTile.Triton.FP.Structural.ObservedInputCellsRegression
