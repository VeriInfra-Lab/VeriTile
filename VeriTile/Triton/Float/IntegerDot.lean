import VeriTile.Triton.Float.IntegerDotPlan
import VeriTile.Triton.Float.WordIntegerDot

/-! Concrete contextual integer dot execution. A context selects only finite
instruction topology from scalar coordinates. It cannot inspect operands,
memory, state, or prospective results. Other source expressions and leaves
retain the existing Structural evaluator. -/
namespace VeriTile.Triton.FP.Structural.IntegerDot
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Closed source profiles. The default retains the existing scalar-cast
interpreter. accumulatorNat recognizes only the original accumulator update,
uses required Nat context registers and unreduced coordinates, and leaves
ifThen in the ordinary Structural evaluator. -/
inductive Mode where
  | scalarCast
  | accumulatorNat
  deriving DecidableEq, Repr

/-- Exact scalar context of the supported source add/dot expression. -/
structure Context where
  rows : Nat
  cols : Nat
  rowRegister : RegName
  columnRegister : RegName
  iterationRegister : RegName
  splitAxis : Nat
  plans : (k row col split iteration : Nat) → Plan k
  mode : Mode := .scalarCast

def dotAdd (arithmetic : Arithmetic) (context : Context) {α : Type}
    (s : State α) {m k n : Nat}
    (old : TileIndex [m,n] → Int) (a : TileIndex [m,k] → Int)
    (b : TileIndex [k,n] → Int) : TileIndex [m,n] → Int :=
  fun i=>
    let pm:=((s.regs .int [] context.rowRegister).getD (fun _=>0) PUnit.unit).toNat
    let pn:=((s.regs .int [] context.columnRegister).getD (fun _=>0) PUnit.unit).toNat
    let t:=(s.regs .nat [] context.iterationRegister).getD (fun _=>0) PUnit.unit
    (context.plans k ((pm*m+i.1.val)%context.rows) ((pn*n+i.2.1.val)%context.cols) (s.pids context.splitAxis) t).evalInt arithmetic
      (fun l=>BitVec.ofInt 8 (a (i.1,l,PUnit.unit)))
      (fun l=>BitVec.ofInt 8 (b (l,i.2.1,PUnit.unit))) (old i)

/-- Only the recognized source integer add/dot node uses the finite plan.
All operands come from the current structural expression evaluator. -/
def addExpression {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context)
    (bc : Broadcast leftShape rightShape shape) (old : Op .int leftShape)
    (right : Op .int rightShape) (s : State α) : Option (TileIndex shape → Int) :=
  match right with
  | .dotInt (batch:=batch) (M:=m) (N:=n) a b options=>
    match batch with
    | []=>if same : shape=[m,n] then do
        let oldValue←evalOp M none old s
        let left←evalOp M none a s
        let right←evalOp M none b s
        let carry : TileIndex [m,n] → Int:=fun i=>oldValue (bc.leftIndex (same.symm ▸ i))
        return fun i=>dotAdd arithmetic context s carry left right (same ▸ i)
      else evalOp M none (.add .int bc old (.dotInt a b options)) s
    | _::_=>evalOp M none (.add .int bc old (.dotInt a b options)) s
  | other=>evalOp M none (.add .int bc old other) s

/-- The accumulator profile requires all three Nat registers, just like its
source executor. Missing registers fail; coordinates are not reduced modulo a
matrix shape. Actual operands and carry come from this source expression. -/
def accumulateNat {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) {m k n : Nat}
    (left : Op .int [m,k]) (right : Op .int [k,n]) (old : Op .int [m,n])
    (s : State α) : Option (TileIndex [m,n] → Int) := do
  let xs←evalOp M none left s
  let ys←evalOp M none right s
  let previous←evalOp M none old s
  let pm←s.regs .nat [] context.rowRegister
  let pn←s.regs .nat [] context.columnRegister
  let iteration←s.regs .nat [] context.iterationRegister
  return fun i=>(context.plans k (pm PUnit.unit*m+i.1.val) (pn PUnit.unit*n+i.2.1.val)
    (s.pids context.splitAxis) (iteration PUnit.unit)).evalInt arithmetic
      (fun j=>BitVec.ofInt 8 (xs (i.1,j,PUnit.unit)))
      (fun j=>BitVec.ofInt 8 (ys (j,i.2.1,PUnit.unit))) (previous i)

/-- Exact accumulator += dot recognition, including destination, old-register,
broadcast, batch and options. Other expressions retain Structural behavior. -/
def accumulatorStep {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) (statement : ComputeStmt)
    (s : State α) : Option (State α) :=
  match statement with
  | .assign .int _ name (.alg (.add .int broadcast old rhs))=>
    match rhs with
    | .dotInt (batch:=batch) left right options=>
      match batch with
      | []=>
        match broadcast with
        | .consSame (.consSame .nil)=>
          match old with
          | .ref .int _ oldName=>
            if name="accumulator" ∧ oldName="accumulator" ∧ options={} then do
              let value←accumulateNat M arithmetic context left right old s
              return s.setReg name .int _ value
            else FP.Structural.step M statement s
          | _=>FP.Structural.step M statement s
        | _=>FP.Structural.step M statement s
      | _=>FP.Structural.step M statement s
    | _=>FP.Structural.step M statement s
  | _=>FP.Structural.step M statement s

def arithmeticStep {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) : ComputeStmt → State α → Option (State α)
  | .assign .int shape name (.alg (.add .int bc old right)),s=>
      match context.mode with
      | .scalarCast=>do
          let value←addExpression M arithmetic context bc old right s
          return s.setReg name .int shape value
      | .accumulatorNat=>accumulatorStep M arithmetic context
          (.assign .int shape name (.alg (.add .int bc old right))) s
  | statement,s=>FP.Structural.step M statement s

mutual
def step {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) : ComputeStmt → State α → Option (State α)
  | .forRange name start stop stride body,s=>range M arithmetic context name start stop stride body s
  | .ifThenElse cond yes no,s=>do
      let value←evalOp M none cond s
      if value PUnit.unit then run M arithmetic context yes s else run M arithmetic context no s
  | .ifThen cond body,s=>
      match context.mode with
      | .scalarCast=>do
          let value←evalOp M none cond s
          if value PUnit.unit then run M arithmetic context body s else some s
      | .accumulatorNat=>FP.Structural.step M (.ifThen cond body) s
  | statement,s=>arithmeticStep M arithmetic context statement s
termination_by statement _ => (sizeOf statement,0)
decreasing_by
  all_goals simp_wf
  all_goals try omega
  all_goals have : 0<sizeOf name := by cases name;simp
  all_goals omega

def run {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) : List ComputeStmt → State α → Option (State α)
  | [],s=>some s
  | statement::rest,s=>do run M arithmetic context rest (←step M arithmetic context statement s)
termination_by code _ => (sizeOf code,0)
decreasing_by all_goals (simp_wf;omega)

def range {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context)
    (name : RegName) (start stop stride : Nat) (body : List ComputeStmt) : State α → Option (State α)
  | s=>if 0<stride ∧ start<stop then do
      let t←run M arithmetic context body (s.setReg name .nat [] (fun _=>start))
      range M arithmetic context name (start+stride) stop stride body t
    else some s
termination_by _ => (sizeOf body+1,stop-start)
decreasing_by all_goals omega
end

def exec {α : Type} [Inhabited α] (M : Algebra α) (arithmetic : Arithmetic)
    (context : Context) (source : ComputeKernel) (s : State α) :=
  run M arithmetic context source.surfaceBody s

theorem run_append {α : Type} [Inhabited α] (M : Algebra α) (arithmetic : Arithmetic)
    (context : Context) (left right : List ComputeStmt) (s : State α) :
    run M arithmetic context (left++right) s=
      (run M arithmetic context left s).bind (run M arithmetic context right) := by
  induction left generalizing s with
  | nil=>simp [run]
  | cons x xs ih=>
    simp only [List.cons_append,run,ih]
    cases step M arithmetic context x s <;> rfl


abbrev step_unfold := @step.eq_def
abbrev run_unfold := @run.eq_def
abbrev range_unfold := @range.eq_def
end
end VeriTile.Triton.FP.Structural.IntegerDot
