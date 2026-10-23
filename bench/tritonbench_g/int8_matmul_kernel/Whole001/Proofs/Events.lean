import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Grid
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 8192

inductive Event (α β : Type) : Type → Type where
  | read (p : Address) : Event α β α
  | cast (x : α) : Event α β Byte
  | packed (x : α) : Event α β Byte
  | zero : Event α β β
  | instruction (kind : Instruction) (xs ys : Fin 32 → Byte) (seed : β) : Event α β β
  | store (p : Address) (x : β) : Event α β Unit

def writeWord (m : Memory α) (p : Address) (x : α) : Memory α := fun q => if q=p then x else m q

def Event.eval (a : Arithmetic α β) : Event α β γ → Memory α → γ × Memory α
  | .read p,m => (m p,m)
  | .cast x,m => (a.cast x,m)
  | .packed x,m => (a.packed x,m)
  | .zero,m => (a.zero,m)
  | .instruction kind xs ys z,m => (a.instruction kind xs ys z,m)
  | .store p x,m => ((),writeWord m p (a.encode x))

inductive Code (α β : Type) : Type → Type 1 where
  | pure (x : γ) : Code α β γ
  | event (e : Event α β γ) (next : γ → Code α β δ) : Code α β δ

def Code.bind (c : Code α β γ) (f : γ → Code α β δ) : Code α β δ :=
  match c with
  | .pure x => f x
  | .event e next => .event e (fun x => (next x).bind f)
instance : Monad (Code α β) where
  pure := Code.pure
  bind := Code.bind

def emit (e : Event α β γ) : Code α β γ := .event e .pure

def Code.eval (a : Arithmetic α β) (c : Code α β γ) (m : Memory α) : γ × Memory α :=
  match c with
  | .pure x => (x,m)
  | .event e next => let (x,m') := e.eval a m; (next x).eval a m'

@[simp] theorem eval_pure (a : Arithmetic α β) (x : γ) (m : Memory α) :
    (Code.pure x).eval a m=(x,m) := rfl
@[simp] theorem eval_emit (a : Arithmetic α β) (e : Event α β γ) (m : Memory α) :
    (emit e).eval a m=e.eval a m := by cases e <;> rfl
@[simp] theorem eval_bind (a : Arithmetic α β) (c : Code α β γ) (f : γ → Code α β δ) (m : Memory α) :
    (c.bind f).eval a m=(f (c.eval a m).1).eval a (c.eval a m).2 := by
  induction c generalizing m with
  | pure x => rfl
  | event e next ih => simp only [Code.bind,Code.eval]; exact ih _ _ _

/-- Typed scalar event boundary. Numeric events expose one conversion or one
fixed K32 ISA lane, never a whole dot, program, launch, or output equation. -/
structure LocalEventABI (α β σ : Type) (a : Arithmetic α β) where
  observe : σ → Memory α
  step : {γ : Type} → Event α β γ → σ → γ × σ
  stepLaw : ∀ (γ : Type) (e : Event α β γ) (state : σ),
    ((step e state).1,observe (step e state).2)=e.eval a (observe state)

def Code.native (abi : LocalEventABI α β σ a) (c : Code α β γ) (state : σ) : γ × σ :=
  match c with
  | .pure x => (x,state)
  | .event e next => let (x,state') := abi.step e state; (next x).native abi state'

theorem Code.native_refines (abi : LocalEventABI α β σ a) (c : Code α β γ) (state : σ) :
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

def gatherCode : {n : Nat} → (Fin n → Code α β γ) → Code α β (Fin n → γ)
  | 0,_ => pure Fin.elim0
  | n+1,f => do
      let x ← f 0
      let xs ← gatherCode (fun i : Fin n => f i.succ)
      pure (Fin.cases x xs)

theorem eval_gather (a : Arithmetic α β) (f : Fin n → Code α β γ) (value : Fin n → γ)
    (m : Memory α) (h : ∀ i,(f i).eval a m=(value i,m)) :
    (gatherCode f).eval a m=(value,m) := by
  induction n with
  | zero =>
    simp only [gatherCode,pure,eval_pure]
    congr 1
    funext i
    exact Fin.elim0 i
  | succ n ih =>
    simp only [gatherCode,bind,eval_bind,h,Prod.fst,Prod.snd]
    rw [ih (fun i => f i.succ) (fun i => value i.succ) (fun i => h i.succ)]
    simp only [Prod.fst,Prod.snd,pure,eval_pure]
    congr 1
    funext i
    exact Fin.cases rfl (fun _ => rfl) i

def foldCode (f : Nat → γ → Code α β γ) : Nat → γ → Code α β γ
  | 0,z => pure z
  | n+1,z => do
      let v ← foldCode f n z
      f n v

theorem eval_foldCode (a : Arithmetic α β) (f : Nat → γ → Code α β γ) (g : Nat → γ → γ)
    (m : Memory α) (h : ∀ i z,(f i z).eval a m=(g i z,m)) (n : Nat) (z : γ) :
    (foldCode f n z).eval a m=(foldN g n z,m) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldCode,bind,eval_bind,ih,Prod.fst,Prod.snd,h,foldN]

def aCode (s : Shape) (b : Buffers) (c : Fin 15) (f j lane m : Nat) : Code α β Byte :=
  if lane<s.k-(f*s.steps c+j)*(configs c).bk then do
    let x ← emit (.read (aAddress b m ((f*s.steps c+j)*(configs c).bk+lane)))
    emit (.cast x)
  else pure 0

def bCode (s : Shape) (b : Buffers) (c : Fin 15) (j lane n : Nat) : Code α β Byte :=
  if lane<s.k then do
    let x ← emit (.read (bAddress b (j*(configs c).bk+lane) n))
    emit (.packed x)
  else pure 0

@[simp] theorem eval_aCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (f j lane m : Nat) (memory : Memory α) :
    (aCode s b c f j lane m).eval a memory=(sourceA a s b c f j lane m memory,memory) := by
  unfold aCode sourceA
  split_ifs <;> rfl
@[simp] theorem eval_bCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (j lane n : Nat) (memory : Memory α) :
    (bCode s b c j lane n).eval a memory=(sourceB a s b c j lane n memory,memory) := by
  unfold bCode sourceB
  split_ifs <;> rfl

def blockCode (s : Shape) (b : Buffers) (c : Fin 15) (f j m n : Nat) (z : β) : Code α β β :=
  foldCode (fun group seed => do
    let xs ← gatherCode (fun lane : Fin 32 => aCode s b c f j (group*32+lane.val) m)
    let ys ← gatherCode (fun lane : Fin 32 => bCode s b c j (group*32+lane.val) n)
    emit (.instruction (selectedInstruction c) xs (fun lane => weightByte (ys lane) f) seed))
    ((configs c).bk/32) z

@[simp] theorem eval_blockCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (f j m n : Nat) (z : β) (memory : Memory α) :
    (blockCode s b c f j m n z).eval a memory=(sourceBlock a s b c f j m n memory z,memory) := by
  simp only [blockCode,sourceBlock]
  apply eval_foldCode
  intro group seed
  simp only [bind,eval_bind,
    eval_gather a _ _ memory (fun i => eval_aCode a _ _ _ _ _ _ _ _),
    eval_gather a _ _ memory (fun i => eval_bCode a _ _ _ _ _ _ _),
    Prod.fst,Prod.snd,eval_emit,Event.eval]

def accumulationCode (s : Shape) (b : Buffers) (c : Fin 15) (m n : Nat) : Code α β β := do
  let zero ← emit .zero
  foldCode (fun f z => foldCode (fun j seed => blockCode s b c f j m n seed) (s.steps c) z) 4 zero

@[simp] theorem eval_accumulationCode (a : Arithmetic α β) (s : Shape) (b : Buffers) (c : Fin 15)
    (m n : Nat) (memory : Memory α) :
    (accumulationCode s b c m n).eval a memory=(sourceAccumulation a s b c memory m n,memory) := by
  simp only [accumulationCode,sourceAccumulation,bind,eval_bind,eval_emit,Event.eval,Prod.fst,Prod.snd]
  apply eval_foldCode
  intro f z
  exact eval_foldCode a _ _ memory (fun j seed => eval_blockCode a s b c f j m n seed memory) _ z

#axiomsClean Code.native_refines
#axiomsClean eval_aCode
#axiomsClean eval_bCode
#axiomsClean eval_blockCode
#axiomsClean eval_accumulationCode
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
