import bench.tritonbench_g.fused_recurrent_retention.Proofs.Memory
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

inductive Event (α : Type) : Type → Type where
  | read (p : Address) : Event α α
  | zero : Event α α
  | one : Event α α
  | negFive : Event α α
  | fromNat (n : Nat) : Event α α
  | scale (dk : Nat) : Event α α
  | add (x y : α) : Event α α
  | sub (x y : α) : Event α α
  | mul (x y : α) : Event α α
  | fma (x y z : α) : Event α α
  | exp2 (x : α) : Event α α
  | store (p : Address) (x : α) : Event α Unit
  | barrier : Event α Unit

def Event.eval (a : Arithmetic α) : Event α β → Memory α → β × Memory α
  | .read p,m => (m p,m)
  | .zero,m => (a.zero,m)
  | .one,m => (a.one,m)
  | .negFive,m => (a.negFive,m)
  | .fromNat n,m => (a.fromNat n,m)
  | .scale dk,m => (a.scale dk,m)
  | .add x y,m => (a.add x y,m)
  | .sub x y,m => (a.sub x y,m)
  | .mul x y,m => (a.mul x y,m)
  | .fma x y z,m => (a.fma x y z,m)
  | .exp2 x,m => (a.exp2 x,m)
  | .store p x,m => ((),writeWord m p x)
  | .barrier,m => ((),m)

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

def Code.eval (a : Arithmetic α) (c : Code α β) (m : Memory α) : β × Memory α :=
  match c with
  | .pure x => (x,m)
  | .event e next => let (x,m') := e.eval a m; (next x).eval a m'

@[simp] theorem eval_pure (a : Arithmetic α) (x : β) (m : Memory α) :
    (Code.pure x).eval a m=(x,m) := rfl
@[simp] theorem eval_emit (a : Arithmetic α) (e : Event α β) (m : Memory α) :
    (emit e).eval a m=e.eval a m := by cases e <;> rfl
@[simp] theorem eval_bind (a : Arithmetic α) (c : Code α β) (f : β → Code α γ) (m : Memory α) :
    (c.bind f).eval a m=(f (c.eval a m).1).eval a (c.eval a m).2 := by
  induction c generalizing m with
  | pure x => rfl
  | event e next ih => simp only [Code.bind,Code.eval]; exact ih _ _ _

/-- One scalar event at a time. Numerical ports have arity at most three;
load/store events expose the current typed memory. Physical compiler/runtime
binding and barrier convergence remain ordinary execution contracts. -/
structure LocalEventABI (α σ : Type) (a : Arithmetic α) where
  observe : σ → Memory α
  step : {β : Type} → Event α β → σ → β × σ
  stepLaw : ∀ (β : Type) (e : Event α β) (st : σ),
    ((step e st).1,observe (step e st).2)=e.eval a (observe st)

def Code.native (abi : LocalEventABI α σ a) (c : Code α β) (st : σ) : β × σ :=
  match c with
  | .pure x => (x,st)
  | .event e next => let (x,st') := abi.step e st; (next x).native abi st'

theorem Code.native_refines (abi : LocalEventABI α σ a) (c : Code α β) (st : σ) :
    ((c.native abi st).1,abi.observe (c.native abi st).2)=c.eval a (abi.observe st) := by
  induction c generalizing st with
  | pure x => rfl
  | event e next ih =>
    have h := abi.stepLaw _ e st
    have hx := congrArg Prod.fst h
    have hm := congrArg Prod.snd h
    simp only [Prod.fst,Prod.snd] at hx hm
    simp only [Code.native,Code.eval]
    rw [ih,hm,hx]

def foldCode (f : Nat → β → Code α β) : Nat → β → Code α β
  | 0,z => pure z
  | n+1,z => do
    let v ← foldCode f n z
    f n v

theorem eval_foldCode (a : Arithmetic α) (f : Nat → β → Code α β) (g : Nat → β → β)
    (m : Memory α) (h : ∀ i z,(f i z).eval a m=(g i z,m)) (n : Nat) (z : β) :
    (foldCode f n z).eval a m=(foldN g n z,m) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldCode,bind,eval_bind,ih,Prod.fst,Prod.snd,h,foldN]

def DotPlan.code (xs ys : Nat → Code α α) : DotPlan → Code α α
  | .product i => do
    let x ← xs i
    let y ← ys i
    emit (.mul x y)
  | .add l r => do
    let x ← l.code xs ys
    let y ← r.code xs ys
    emit (.add x y)
  | .fma i tail => do
    let x ← xs i
    let y ← ys i
    let z ← tail.code xs ys
    emit (.fma x y z)

theorem DotPlan.eval_code (a : Arithmetic α) (p : DotPlan) (xs ys : Nat → Code α α)
    (x y : Nat → α) (m : Memory α) (hx : ∀ i,(xs i).eval a m=(x i,m))
    (hy : ∀ i,(ys i).eval a m=(y i,m)) :
    (p.code xs ys).eval a m=(p.eval a x y,m) := by
  induction p with
  | product i => simp [DotPlan.code,DotPlan.eval,bind,eval_bind,hx,hy,Event.eval]
  | add l r hl hr => simp [DotPlan.code,DotPlan.eval,bind,eval_bind,hl,hr,Event.eval]
  | fma i tail ih => simp [DotPlan.code,DotPlan.eval,bind,eval_bind,hx,hy,ih,Event.eval]

def SumPlan.code (xs : Nat → Code α α) : SumPlan → Code α α
  | .zero => emit .zero
  | .input i => xs i
  | .add l r => do
    let x ← l.code xs
    let y ← r.code xs
    emit (.add x y)

theorem SumPlan.eval_code (a : Arithmetic α) (p : SumPlan) (xs : Nat → Code α α)
    (x : Nat → α) (m : Memory α) (hx : ∀ i,(xs i).eval a m=(x i,m)) :
    (p.code xs).eval a m=(p.eval a x,m) := by
  induction p with
  | zero => rfl
  | input i => exact hx i
  | add l r hl hr => simp [SumPlan.code,SumPlan.eval,bind,eval_bind,hl,hr,Event.eval]

def loadCode (active : Bool) (p : Address) : Code α α :=
  if active then emit (.read p) else emit .zero

@[simp] theorem eval_loadCode (a : Arithmetic α) (active : Bool) (p : Address) (m : Memory α) :
    (loadCode active p).eval a m=((if active then m p else a.zero),m) := by
  cases active <;> rfl

def decayCode (head : Nat) : Code α α := do
  let h ← emit (.fromNat head)
  let one ← emit .one
  let h ← emit (.mul h one)
  let n ← emit .negFive
  let arg ← emit (.sub n h)
  let e ← emit (.exp2 arg)
  let one ← emit .one
  emit (.sub one e)

@[simp] theorem eval_decayCode (a : Arithmetic α) (head : Nat) (m : Memory α) :
    (decayCode head).eval a m=(decay a head,m) := by rfl

def stateCode (s : Shape) (b : InputBuffers) (seed : Bool) (h k v n : Nat) : Code α α := do
  let zero ← emit .zero
  let beta ← decayCode (h%s.H)
  let initial ← if seed then do
      let x ← loadCode (decide (k<s.DK ∧ v<s.DV)) (b.seedAddress s h k v)
      emit (.add zero x)
    else pure zero
  foldCode (fun t old => do
    let key ← loadCode (decide (k<s.DK)) (b.kAddress s h t k)
    let value ← loadCode (decide (v<s.DV)) (b.vAddress s h t v)
    let product ← emit (.mul key value)
    emit (.fma beta old product)) n initial

@[simp] theorem eval_stateCode (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (seed : Bool) (h k v n : Nat) (m : Memory α) :
    (stateCode s b seed h k v n).eval a m=(state a s (b.observe s m) seed h k v n,m) := by
  cases seed <;>
    simp only [stateCode,bind,eval_bind,eval_emit,Event.eval,eval_decayCode,Prod.fst,Prod.snd,
      Bool.false_eq_true,↓reduceIte,eval_loadCode,pure,eval_pure]
  all_goals
    unfold state
    simp only [Bool.false_eq_true,↓reduceIte,stateLoad,InputBuffers.observe,decide_eq_true_eq]
    apply eval_foldCode
    intro t old
    simp [bind,eval_bind,Event.eval,keyLoad,valueLoad,stateLoad,InputBuffers.observe]

def reverseCode (s : Shape) (b : InputBuffers) (final : Bool) (h k v n : Nat) : Code α α := do
  let beta ← decayCode (h%s.H)
  let scale ← emit (.scale s.DK)
  let initial ← if final then loadCode (decide (k<s.DK ∧ v<s.DV)) (b.finalAddress s h k v)
    else emit .zero
  foldCode (fun r old => do
    let t := s.T-1-r
    let q ← loadCode (decide (k<s.DK)) (b.qAddress s h t k)
    let q ← emit (.mul q scale)
    let d ← loadCode (decide (v<s.DV)) (b.doAddress s h t v)
    let z ← emit (.fma q d old)
    emit (.mul z beta)) n initial

@[simp] theorem eval_reverseCode (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (final : Bool) (h k v n : Nat) (m : Memory α) :
    (reverseCode s b final h k v n).eval a m=(reverseCarry a s (b.observe s m) final h k v n,m) := by
  cases final <;>
    simp only [reverseCode,bind,eval_bind,eval_emit,Event.eval,eval_decayCode,Prod.fst,Prod.snd,
      Bool.false_eq_true,↓reduceIte,eval_loadCode]
  all_goals
    unfold reverseCarry
    simp only [Bool.false_eq_true,↓reduceIte,stateLoad,InputBuffers.observe,decide_eq_true_eq]
    apply eval_foldCode
    intro r old
    simp [bind,eval_bind,Event.eval,keyLoad,valueLoad,stateLoad,InputBuffers.observe]

def adjointCode (s : Shape) (b : InputBuffers) (final : Bool) (h t k v : Nat) : Code α α := do
  let old ← reverseCode s b final h k v (s.T-1-t)
  let q ← loadCode (decide (k<s.DK)) (b.qAddress s h t k)
  let scale ← emit (.scale s.DK)
  let q ← emit (.mul q scale)
  let d ← loadCode (decide (v<s.DV)) (b.doAddress s h t v)
  emit (.fma q d old)

@[simp] theorem eval_adjointCode (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (final : Bool) (h t k v : Nat) (m : Memory α) :
    (adjointCode s b final h t k v).eval a m=(adjoint a s (b.observe s m) final h t k v,m) := by
  simp [adjointCode,adjoint,bind,eval_bind,Event.eval,keyLoad,valueLoad,InputBuffers.observe]

#axiomsClean Code.native_refines
#axiomsClean DotPlan.eval_code
#axiomsClean SumPlan.eval_code
#axiomsClean eval_stateCode
#axiomsClean eval_reverseCode
#axiomsClean eval_adjointCode
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
