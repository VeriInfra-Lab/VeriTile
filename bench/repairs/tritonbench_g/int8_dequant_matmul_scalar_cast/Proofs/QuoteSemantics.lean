import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Contract
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Representation of an integer accumulator in the source's discrete
register channel. Its operations are individual scalar/fixed K32 leaves. -/
structure IntegerModel (I F H : Type) where
  ops : Ops I F H
  encode : Int → I
  decode : I → Int
  roundtrip : ∀ x,encode (decode x)=x
  zero : decode ops.izero=0
  plans : (k row col split iteration : Nat) → DotPlan k

/-- Source-local INT32 dot-plus-add. Fusion is selected at this expression;
the actual old accumulator is passed to the fixed instruction plan. This is
an implementation, not an assumed result of a variable-length dot. -/
def dotAdd (q : IntegerModel I F H) (rows cols : Nat) {α : Type}
    (s : State α) {m k n : Nat}
    (old : TileIndex [m,n] → Int) (a : TileIndex [m,k] → Int)
    (b : TileIndex [k,n] → Int) : TileIndex [m,n] → Int :=
  fun i=>
    let pm:=((s.regs .int [] "pid_m").getD (fun _=>0) PUnit.unit).toNat
    let pn:=((s.regs .int [] "pid_n").getD (fun _=>0) PUnit.unit).toNat
    let t:=(s.regs .nat [] "k").getD (fun _=>0) PUnit.unit
    q.decode ((q.plans k ((pm*m+i.1.val)%rows) ((pn*n+i.2.1.val)%cols) (s.pids 1) t).eval q.ops
      (fun l=>BitVec.ofInt 8 (a (i.1,l,PUnit.unit)))
      (fun l=>BitVec.ofInt 8 (b (l,i.2.1,PUnit.unit))) (q.encode (old i)))

/-- All source nodes use the shared interpreter except this exact integer
dot/add expression. The shared mathematical Int addition cannot represent a
selected saturating or wrapping fused instruction, so this local numeric
adapter evaluates that syntax with the fixed instruction model above. -/
def addExpression {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat)
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
        return fun i=>dotAdd q rows cols s carry left right (same ▸ i)
      else evalOp M none (.add .int bc old (.dotInt a b options)) s
    | _::_=>evalOp M none (.add .int bc old (.dotInt a b options)) s
  | other=>evalOp M none (.add .int bc old other) s

def arithmeticStep {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) : ComputeStmt → State α → Option (State α)
  | .assign .int shape name (.alg (.add .int bc old right)),s=>do
      let value←addExpression M q rows cols bc old right s
      return s.setReg name .int shape value
  | statement,s=>FP.Structural.step M statement s

mutual
def step {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) : ComputeStmt → State α → Option (State α)
  | .forRange name start stop stride body,s=>range M q rows cols name start stop stride body s
  | .ifThenElse cond yes no,s=>do
      let value←evalOp M none cond s
      if value PUnit.unit then run M q rows cols yes s else run M q rows cols no s
  | .ifThen cond body,s=>do
      let value←evalOp M none cond s
      if value PUnit.unit then run M q rows cols body s else some s
  | statement,s=>arithmeticStep M q rows cols statement s
termination_by statement _ => (sizeOf statement,0)
decreasing_by
  all_goals simp_wf
  all_goals try omega
  all_goals have : 0<sizeOf name := by cases name;simp
  all_goals omega

def run {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) : List ComputeStmt → State α → Option (State α)
  | [],s=>some s
  | statement::rest,s=>do run M q rows cols rest (←step M q rows cols statement s)
termination_by code _ => (sizeOf code,0)
decreasing_by all_goals (simp_wf;omega)

def range {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat)
    (name : RegName) (start stop stride : Nat) (body : List ComputeStmt) : State α → Option (State α)
  | s=>if 0<stride ∧ start<stop then do
      let t←run M q rows cols body (s.setReg name .nat [] (fun _=>start))
      range M q rows cols name (start+stride) stop stride body t
    else some s
termination_by _ => (sizeOf body+1,stop-start)
decreasing_by all_goals omega
end

def execute {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (rows cols : Nat) (source : ComputeKernel) (s : State α) :=
  run M q rows cols source.surfaceBody s

theorem run_append {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (rows cols : Nat) (left right : List ComputeStmt) (s : State α) :
    run M q rows cols (left++right) s=
      (run M q rows cols left s).bind (run M q rows cols right) := by
  induction left generalizing s with
  | nil=>simp [run]
  | cons x xs ih=>
    simp only [List.cons_append,run,ih]
    cases step M q rows cols x s <;> rfl

#axiomsClean run_append
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
