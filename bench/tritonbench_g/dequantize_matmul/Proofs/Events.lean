import bench.tritonbench_g.dequantize_matmul.Proofs.FPReference

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- Every memory effect is a single typed load or single-word store. Numeric
operands are values from earlier events, not whole output/reduction oracles. -/
inductive Event (α : Type) : Type → Type where
  | read8 (address : Address) : Event α (BitVec 8)
  | read32 (address : Address) : Event α α
  | zero : Event α α
  | signed8 (q : BitVec 8) : Event α α
  | mul (x y : α) : Event α α
  | add (x y : α) : Event α α
  | fma (x y z : α) : Event α α
  | tf32 (mode : TF32Mode) (x : α) : Event α α
  | mma8 (kind : MMAKind) (xs ys : Fin 8 → α) (seed : α) : Event α α
  | store32 (address : Address) (x : α) : Event α Unit

def writeWord (m : Memory (Cell α)) (p : Address) (x : α) : Memory (Cell α) :=
  fun q => if q=p then .fp32 x else m q

def Event.eval (a : Arithmetic α) : Event α β → Memory (Cell α) → β × Memory (Cell α)
  | .read8 p,m => ((m p).signed,m)
  | .read32 p,m => (a.load ((m p).float a.zero),m)
  | .zero,m => (a.zero,m)
  | .signed8 q,m => (a.signed8 q,m)
  | .mul x y,m => (a.mul x y,m)
  | .add x y,m => (a.add x y,m)
  | .fma x y z,m => (a.fma x y z,m)
  | .tf32 mode x,m => (a.tf32 mode x,m)
  | .mma8 kind xs ys z,m => (a.mma8 kind xs ys z,m)
  | .store32 p x,m => ((),writeWord m p (a.store x))

inductive Code (α : Type) : Type → Type 1 where
  | pure (x : β) : Code α β
  | event (e : Event α β) (next : β → Code α γ) : Code α γ

def Code.bind (c : Code α β) (f : β → Code α γ) : Code α γ :=
  match c with
  | .pure x => f x
  | .event e next => .event e (fun x => (next x).bind f)
instance : Monad (Code α) where
  pure := Code.pure
  bind := Code.bind

def emit (e : Event α β) : Code α β := .event e .pure

def Code.eval (a : Arithmetic α) (c : Code α β) (m : Memory (Cell α)) : β × Memory (Cell α) :=
  match c with
  | .pure x => (x,m)
  | .event e next => let (x,m') := e.eval a m; (next x).eval a m'

@[simp] theorem eval_pure (a : Arithmetic α) (x : β) (m : Memory (Cell α)) :
    (Code.pure x).eval a m=(x,m) := rfl
@[simp] theorem eval_emit (a : Arithmetic α) (e : Event α β) (m : Memory (Cell α)) :
    (emit e).eval a m=e.eval a m := by cases e <;> rfl
@[simp] theorem eval_bind (a : Arithmetic α) (c : Code α β) (f : β → Code α γ)
    (m : Memory (Cell α)) :
    (c.bind f).eval a m=(f (c.eval a m).1).eval a (c.eval a m).2 := by
  induction c generalizing m with
  | pure x => rfl
  | event e next ih =>
    simp only [Code.bind,Code.eval]
    exact ih _ _ _

/-- A local event ABI. Each primitive event is bound to the same selected
implementation's step; it exposes one value and that step's memory effect.
There is deliberately no whole-program simulation or final-memory field. -/
structure LocalEventABI (α σ : Type) (a : Arithmetic α) where
  observe : σ → Memory (Cell α)
  step : {β : Type} → Event α β → σ → β × σ
  stepLaw : ∀ (β : Type) (e : Event α β) (state : σ),
    ((step e state).1,observe (step e state).2)=e.eval a (observe state)

def Code.native (abi : LocalEventABI α σ a) (c : Code α β) (state : σ) : β × σ :=
  match c with
  | .pure x => (x,state)
  | .event e next => let (x,state') := abi.step e state; (next x).native abi state'

/-- Derived by structural induction over the actual chosen event program. -/
theorem Code.native_refines (abi : LocalEventABI α σ a) (c : Code α β) (state : σ) :
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

/-- No opaque loads occur inside the plan: each source operand is read through
an explicit event from the current matrix/scratch location. -/
def planCode (s : Shape) (b : Buffers) (r n : Nat) : Plan s.K → Code α α
  | .zero => emit .zero
  | .product k => do
      let x ← emit (.read32 (b.a.addr (r*s.K+k.val)))
      let y ← emit (.read32 (b.fpb.addr (k.val*s.N+n)))
      emit (.mul x y)
  | .fma k seed => do
      let z ← planCode s b r n seed
      let x ← emit (.read32 (b.a.addr (r*s.K+k.val)))
      let y ← emit (.read32 (b.fpb.addr (k.val*s.N+n)))
      emit (.fma x y z)
  | .add left right => do
      let x ← planCode s b r n left
      let y ← planCode s b r n right
      emit (.add x y)
  | .mma8 kind conversion lanes seed => do
      let z ← planCode s b r n seed
      let xs ← lanesCode conversion (fun k => b.a.addr (r*s.K+k.val)) lanes
      let ys ← lanesCode conversion (fun k => b.fpb.addr (k.val*s.N+n)) lanes
      emit (.mma8 kind xs ys z)
where
  lanesCode (mode : TF32Mode) (loc : Fin s.K → Address) (lanes : Fin 8 → Option (Fin s.K)) : Code α (Fin 8 → α) := do
    let x0 ← laneCode mode loc (lanes 0); let x1 ← laneCode mode loc (lanes 1)
    let x2 ← laneCode mode loc (lanes 2); let x3 ← laneCode mode loc (lanes 3)
    let x4 ← laneCode mode loc (lanes 4); let x5 ← laneCode mode loc (lanes 5)
    let x6 ← laneCode mode loc (lanes 6); let x7 ← laneCode mode loc (lanes 7)
    pure (fun i => ![x0,x1,x2,x3,x4,x5,x6,x7] i)
  laneCode (mode : TF32Mode) (loc : Fin s.K → Address) (lane : Option (Fin s.K)) : Code α α := do
    let x ← match lane with
      | none => emit .zero
      | some k => emit (.read32 (loc k))
    emit (.tf32 mode x)

set_option maxRecDepth 4096 in
@[simp] theorem eval_laneCode (a : Arithmetic α) (s : Shape) (mode : TF32Mode) (loc : Fin s.K → Address)
    (lane : Option (Fin s.K)) (m : Memory (Cell α)) :
    (planCode.laneCode s mode loc lane).eval a m=
      (a.tf32 mode ((lane.map (fun k => a.load ((m (loc k)).float a.zero))).getD a.zero),m) := by
  cases lane <;> simp [planCode.laneCode,bind,Code.bind,Code.eval,emit,Event.eval,pure]

set_option maxRecDepth 4096 in
@[simp] theorem eval_lanesCode (a : Arithmetic α) (s : Shape) (mode : TF32Mode) (loc : Fin s.K → Address)
    (lanes : Fin 8 → Option (Fin s.K)) (m : Memory (Cell α)) :
    (planCode.lanesCode s mode loc lanes).eval a m=
      ((fun i => ((lanes i).map (fun k => a.tf32 mode (a.load ((m (loc k)).float a.zero)))).getD (a.tf32 mode a.zero)),m) := by
  simp only [planCode.lanesCode,bind,eval_bind,eval_laneCode,Prod.fst,Prod.snd,pure,eval_pure]
  congr 1
  funext i
  fin_cases i <;> simp <;> cases lanes _ <;> rfl

@[simp] theorem eval_planCode (a : Arithmetic α) (s : Shape) (b : Buffers)
    (r n : Nat) (p : Plan s.K) (m : Memory (Cell α)) :
    (planCode s b r n p).eval a m=
      (p.eval a (fun k => (m (b.a.addr (r*s.K+k.val))).float a.zero)
        (fun k => (m (b.fpb.addr (k.val*s.N+n))).float a.zero),m) := by
  induction p with
  | zero => rfl
  | product k => rfl
  | fma k seed ih =>
    simp [planCode,bind,eval_bind,ih,Event.eval,Plan.eval]
  | add left right il ir =>
    simp [planCode,bind,eval_bind,il,ir,Event.eval,Plan.eval]
  | mma8 kind conversion lanes seed ih =>
    simp only [planCode,bind,eval_bind,ih,Prod.fst,Prod.snd,eval_lanesCode,eval_emit,Event.eval,Plan.eval]

def dequantCode (b : Buffers) (s : Shape) (k n : Nat) : Code α Unit := do
  let q ← emit (.read8 (b.b.addr (k*s.N+n)))
  let scale ← emit (.read32 (b.scale.addr n))
  let converted ← emit (.signed8 q)
  let product ← emit (.mul converted scale)
  emit (.store32 (b.fpb.addr (k*s.N+n)) product)
def outputCode (b : Buffers) (s : Shape) (plan : LibraryPlan s) (r n : Nat) : Code α Unit := do
  let value ← planCode s b r n (plan.output r n)
  emit (.store32 (b.out.addr (r*s.N+n)) value)

@[simp] theorem eval_dequantCode (a : Arithmetic α) (b : Buffers) (s : Shape)
    (k n : Nat) (m : Memory (Cell α)) :
    (dequantCode b s k n).eval a m=
      ((),writeWord m (b.fpb.addr (k*s.N+n)) ((dequantValue a s b m k n).float a.zero)) := by rfl
@[simp] theorem eval_outputCode (a : Arithmetic α) (b : Buffers) (s : Shape)
    (plan : LibraryPlan s) (r n : Nat) (m : Memory (Cell α)) :
    (outputCode b s plan r n).eval a m=
      ((),writeWord m (b.out.addr (r*s.N+n)) ((mmValue a s b plan m r n).float a.zero)) := by
  simp [outputCode,bind,eval_bind,eval_planCode,Event.eval,mmValue,Cell.float]

#axiomsClean Code.native_refines
#axiomsClean eval_planCode
#axiomsClean eval_dequantCode
#axiomsClean eval_outputCode
end
end VeriTile.Bench.Repairs.DequantizeMatmul
