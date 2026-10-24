import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Grid

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

inductive Event (I F H : Type) : Type → Type where
  | read8 (p : Address) : Event I F H (BitVec 8)
  | read32 (p : Address) : Event I F H F
  | read16 (p : Address) : Event I F H H
  | fzero : Event I F H F
  | hzero : Event I F H H
  | widen (x : H) : Event I F H F
  | add (x y : F) : Event I F H F
  | absolute (x : F) : Event I F H F
  | maximum (x y : F) : Event I F H F
  | divide (x y : F) : Event I F H F
  | scale127 (x : F) : Event I F H F
  | integer8 (x : F) : Event I F H Byte
  | store8 (p : Address) (x : Byte) : Event I F H Unit
  | store32 (p : Address) (x : F) : Event I F H Unit
  | atomic32 (p : Address) (x : F) : Event I F H Unit
  | izero : Event I F H I
  | product (x y : BitVec 8) : Event I F H I
  | iadd (x y : I) : Event I F H I
  | mma32 (kind : ISA) (xs ys : Fin 32 → BitVec 8) (seed : I) : Event I F H I
  | convert (x : I) : Event I F H F
  | mul (x y : F) : Event I F H F
  | narrow (x : F) : Event I F H H
  | hadd (x y : H) : Event I F H H
  | store (p : Address) (x : H) : Event I F H Unit
  | atomic (p : Address) (x : H) : Event I F H Unit

def Event.eval (a : Ops I F H) : Event I F H β → Memory F H → β × Memory F H
  | .read8 p,m => (Int8MatmulQuantization.read8 m p,m)
  | .read32 p,m => (Int8MatmulQuantization.read32 a.toOps m p,m)
  | .read16 p,m => (Int8MatmulQuantization.read16 a.toOps m p,m)
  | .fzero,m => (a.fzero,m)
  | .hzero,m => (a.hzero,m)
  | .widen x,m => (a.widen x,m)
  | .add x y,m => (a.add x y,m)
  | .absolute x,m => (a.absolute x,m)
  | .maximum x y,m => (a.maximum x y,m)
  | .divide x y,m => (a.divide x y,m)
  | .scale127 x,m => (a.scale127 x,m)
  | .integer8 x,m => (a.integer8 x,m)
  | .store8 p x,m => ((),write8 m p x)
  | .store32 p x,m => ((),write32 m p x)
  | .atomic32 p x,m => ((),write32 m p (a.add (Int8MatmulQuantization.read32 a.toOps m p) x))
  | .izero,m => (a.izero,m)
  | .product x y,m => (a.product8 x y,m)
  | .iadd x y,m => (a.iadd x y,m)
  | .mma32 kind xs ys old,m => (a.mma32 kind xs ys old,m)
  | .convert x,m => (a.toFloat x,m)
  | .mul x y,m => (a.mul x y,m)
  | .narrow x,m => (a.narrow x,m)
  | .hadd x y,m => (a.hadd x y,m)
  | .store p x,m => ((),write16 m p x)
  | .atomic p x,m => ((),write16 m p (a.hadd (Int8MatmulQuantization.read16 a.toOps m p) x))

inductive Code (I F H : Type) : Type → Type 1 where
  | pure (x : β) : Code I F H β
  | event (e : Event I F H β) (next : β → Code I F H γ) : Code I F H γ
def Code.bind (c : Code I F H β) (f : β → Code I F H γ) : Code I F H γ :=
  match c with
  | .pure x => f x
  | .event e next => .event e (fun x => (next x).bind f)
instance : Monad (Code I F H) where
  pure := Code.pure
  bind := Code.bind
def emit (e : Event I F H β) : Code I F H β := .event e .pure
def Code.eval (a : Ops I F H) (c : Code I F H β) (m : Memory F H) : β × Memory F H :=
  match c with
  | .pure x => (x,m)
  | .event e next => let (x,m') := e.eval a m; (next x).eval a m'
@[simp] theorem eval_pure (a : Ops I F H) (x : β) (m : Memory F H) :
    (Code.pure x).eval a m=(x,m) := rfl
@[simp] theorem eval_emit (a : Ops I F H) (e : Event I F H β) (m : Memory F H) :
    (emit e).eval a m=e.eval a m := by cases e <;> rfl
@[simp] theorem eval_bind (a : Ops I F H) (c : Code I F H β) (f : β → Code I F H γ)
    (m : Memory F H) :
    (c.bind f).eval a m=(f (c.eval a m).1).eval a (c.eval a m).2 := by
  induction c generalizing m with
  | pure x => rfl
  | event e next ih => simp only [Code.bind,Code.eval]; exact ih _ _ _

/-- Only individual memory operations, scalar arithmetic, and fixed K32 ISA
results are exposed at the implementation boundary. There is no dot, kernel,
matrix, successful execution, output, or caller-frame hypothesis. -/
structure LocalEventABI (I F H σ : Type) (a : Ops I F H) where
  observe : σ → Memory F H
  step : {β : Type} → Event I F H β → σ → β × σ
  stepLaw : ∀ (β : Type) (e : Event I F H β) (state : σ),
    ((step e state).1,observe (step e state).2)=e.eval a (observe state)
def Code.native (abi : LocalEventABI I F H σ a) (c : Code I F H β) (state : σ) : β × σ :=
  match c with
  | .pure x => (x,state)
  | .event e next => let (x,state') := abi.step e state; (next x).native abi state'
theorem Code.native_refines (abi : LocalEventABI I F H σ a) (c : Code I F H β) (state : σ) :
    ((c.native abi state).1,abi.observe (c.native abi state).2)=c.eval a (abi.observe state) := by
  induction c generalizing state with
  | pure x => rfl
  | event e next ih =>
    have h := abi.stepLaw _ e state
    have hx := congrArg Prod.fst h
    have hm := congrArg Prod.snd h
    simp only [Prod.fst,Prod.snd] at hx hm
    simp only [Code.native,Code.eval]
    rw [ih,hm,hx]

def vectorCode : (n : Nat) → (Fin n → Code I F H β) → Code I F H (Fin n → β)
  | 0,_ => pure Fin.elim0
  | n+1,f => do
      let x ← f 0
      let xs ← vectorCode n (fun i => f i.succ)
      pure (Fin.cases x xs)
theorem eval_vectorCode (a : Ops I F H) (n : Nat) (f : Fin n → Code I F H β)
    (v : Fin n → β) (m : Memory F H) (h : ∀ i,(f i).eval a m=(v i,m)) :
    (vectorCode n f).eval a m=(v,m) := by
  induction n with
  | zero => simp [vectorCode,pure]; congr; funext i; exact Fin.elim0 i
  | succ n ih =>
    simp only [vectorCode,bind,eval_bind,h,Prod.fst,Prod.snd,ih _ _ (fun i => h i.succ),pure,eval_pure]
    congr 1
    funext i
    exact Fin.cases rfl (fun _ => rfl) i

def microCode (aOp : Micro BK) (xs ys : Fin BK → BitVec 8) (old : I) : Code I F H I :=
  match aOp with
  | .scalar k => do
      let product ← emit (.product (xs k) (ys k))
      emit (.iadd old product)
  | .instruction kind lanes => emit (.mma32 kind
      (fun i => ((lanes i).map xs).getD 0) (fun i => ((lanes i).map ys).getD 0) old)
def operationsCode (xs ys : Fin BK → BitVec 8) : List (Micro BK) → I → Code I F H I
  | [],old => pure old
  | op::ops,old => do
      let v ← microCode op xs ys old
      operationsCode xs ys ops v
theorem eval_microCode (a : Ops I F H) (op : Micro BK) (xs ys : Fin BK → BitVec 8)
    (old : I) (m : Memory F H) : (microCode op xs ys old).eval a m=(op.eval a.toOps xs ys old,m) := by
  cases op <;> rfl
theorem eval_operationsCode (a : Ops I F H) (ops : List (Micro BK)) (xs ys : Fin BK → BitVec 8)
    (old : I) (m : Memory F H) :
    (operationsCode xs ys ops old).eval a m=(ops.foldl (fun v op => op.eval a.toOps xs ys v) old,m) := by
  induction ops generalizing old with
  | nil => rfl
  | cons op ops ih => simp only [operationsCode,bind,eval_bind,eval_microCode,Prod.fst,Prod.snd,ih,List.foldl_cons]
def dotCode (p : DotPlan BK) (xs ys : Fin BK → BitVec 8) (old : I) : Code I F H I := do
  if p.fused then operationsCode xs ys p.operations old
  else
    let zero ← emit .izero
    let dot ← operationsCode xs ys p.operations zero
    emit (.iadd old dot)
theorem eval_dotCode (a : Ops I F H) (p : DotPlan BK) (xs ys : Fin BK → BitVec 8)
    (old : I) (m : Memory F H) : (dotCode p xs ys old).eval a m=(p.eval a.toOps xs ys old,m) := by
  by_cases h : p.fused
  · simp [dotCode,h,eval_operationsCode,Int8DequantMatmul.DotPlan.eval]
  · simp [dotCode,h,bind,eval_bind,eval_operationsCode,Event.eval,Int8DequantMatmul.DotPlan.eval]

def scaleCode (precision : Precision) (p : Address) : Code I F H F :=
  match precision with
  | .single => emit (.read32 p)
  | .half => do
      let value ← emit (.read16 p)
      emit (.widen value)
theorem eval_scaleCode (a : Ops I F H) (precision : Precision) (p : Address) (m : Memory F H) :
    (scaleCode precision p).eval a m=(loadScale a precision m p,m) := by
  cases precision <;> rfl

def operandCode (s : Shape) (c : Config) (b : Buffers) (r t z : Nat)
    (left : Bool) (l : Fin c.BK) : Code I F H Byte :=
  let k := sourceK c t z l.val
  if k<s.K then
    emit (.read8 (if left then (b.a,b.aOffset+r*s.K+k) else (b.b,b.bIndex s k r)))
  else pure 0
def accumulationCode (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (r n z : Nat) : Nat → Code I F H I
  | 0 => emit .izero
  | t+1 => do
      let old ← accumulationCode s c b selected r n z t
      let xs ← vectorCode c.BK (operandCode s c b r t z Bool.true)
      let ys ← vectorCode c.BK (operandCode s c b n t z Bool.false)
      dotCode (selected.plan r n z t) xs ys old
theorem eval_accumulationCode (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (r n z t : Nat) (m : Memory F H) :
    (accumulationCode s c b selected r n z t).eval a m=(accumulation a s c b selected m r n z t,m) := by
  have hx t : ∀ l,(operandCode s c b r t z Bool.true l).eval a m=(operandA s c b m r t z l,m) := by
    intro l
    by_cases h : sourceK c t z l.val<s.K <;>
      simp [operandCode,operandA,h,Event.eval,pure,Nat.add_assoc]
  have hy t : ∀ l,(operandCode s c b n t z Bool.false l).eval a m=(operandB s c b m n t z l,m) := by
    intro l
    by_cases h : sourceK c t z l.val<s.K <;>
      simp [operandCode,operandB,h,Event.eval,pure,Nat.add_assoc]
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [accumulationCode,bind,eval_bind,ih,Prod.fst,Prod.snd,
      eval_vectorCode a _ _ _ m (hx t),eval_vectorCode a _ _ _ m (hy t),eval_dotCode,accumulation]

def partialCode (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (r n z : Nat) : Code I F H H := do
  let sa ← scaleCode b.aScalePrecision (b.aScale,b.aScaleOffset+r)
  let sb ← scaleCode b.bScalePrecision (b.bScale,b.bScaleOffset+n)
  let acc ← accumulationCode s c b selected r n z (steps s c)
  let fp ← emit (.convert acc)
  let scaled ← emit (.mul fp sa)
  let value ← emit (.mul scaled sb)
  emit (.narrow value)
theorem eval_partialCode (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (r n z : Nat) (m : Memory F H) :
    (partialCode s c b selected r n z).eval a m=(partialValue a s c b selected m r n z,m) := by
  simp only [partialCode,bind,eval_bind,eval_scaleCode,eval_emit,Event.eval,Prod.fst,Prod.snd,
    eval_accumulationCode,partialValue]

def commitCode (s : Shape) (c : Config) (b : Buffers) (value : H) (e : Commit s c) : Code I F H Unit :=
  match b.outPrecision with
  | .half => if c.S=1 then emit (.store (outputAddress s b e.1) value)
      else emit (.atomic (outputAddress s b e.1) value)
  | .single => do
      let widened ← emit (.widen value)
      if c.S=1 then emit (.store32 (outputAddress s b e.1) widened)
      else emit (.atomic32 (outputAddress s b e.1) widened)
theorem eval_commitCode (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (value : H) (e : Commit s c) (m : Memory F H) :
    (commitCode s c b value e).eval a m=((),commit a s c b value e m) := by
  cases mode : b.outPrecision <;> by_cases single : c.S=1 <;>
    simp [commitCode,mode,single,bind,eval_bind,Event.eval,commit,contribution,outputAdd,
      writeStored,write16,Int8DequantMatmul.write16,write32,readStored]
  all_goals
    cases cell : m (outputAddress s b e.1) with
    | inl x =>
        try simp only [read16,read32,Int8DequantMatmul.read16,Int8DequantMatmul.read32,cell]
        rfl
    | inr x =>
        cases x
        all_goals
          try simp only [read16,read32,Int8DequantMatmul.read16,Int8DequantMatmul.read32,cell]
          rfl

def gridCode (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c) :
    List (Commit s c) → Code I F H Unit
  | [] => pure ()
  | e::es => do
      let value ← partialCode s c b selected e.1.1 e.1.2 e.2
      let _ ← commitCode s c b value e
      gridCode s c b selected es
theorem eval_gridCode (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (events : List (Commit s c)) (m : Memory F H) :
    (gridCode s c b selected events).eval a m=((),gridRun a s c b selected events m) := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    simp only [gridCode,bind,eval_bind,eval_partialCode,eval_commitCode,Prod.fst,Prod.snd,ih,gridRun]

#axiomsClean Code.native_refines
#axiomsClean eval_accumulationCode
#axiomsClean eval_partialCode
#axiomsClean eval_commitCode
#axiomsClean eval_gridCode
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
