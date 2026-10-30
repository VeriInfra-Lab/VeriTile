import bench.tritonbench_g.triton_linear_activation.Proofs.Grid
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 8192

inductive Event (α : Type) : Type → Type where
  | read (mode : Mode) (p : Address) : Event α α
  | literal (l : Literal) : Event α α
  | add (x y : α) : Event α α
  | mul (x y : α) : Event α α
  | fma (x y z : α) : Event α α
  | divFull (x y : α) : Event α α
  | maximum (x y : α) : Event α α
  | tanh (x : α) : Event α α
  | erf (x : α) : Event α α
  | instruction (kind : Instruction) (xs ys : Fin kind.width → α) (seed : α) : Event α α
  | store (mode : Mode) (p : Address) (x : α) : Event α Unit

def writeWord (m : Memory α) (p : Address) (x : α) : Memory α := fun q => if q=p then x else m q

def Event.eval (a : Arithmetic α) : Event α β → Memory α → β × Memory α
  | .read mode p,m => (a.decode mode (m p),m)
  | .literal l,m => (a.literal l,m)
  | .add x y,m => (a.add x y,m)
  | .mul x y,m => (a.mul x y,m)
  | .fma x y z,m => (a.fma x y z,m)
  | .divFull x y,m => (a.divFull x y,m)
  | .maximum x y,m => (a.maximum x y,m)
  | .tanh x,m => (a.tanh x,m)
  | .erf x,m => (a.erf x,m)
  | .instruction kind xs ys z,m => (a.instruction kind xs ys z,m)
  | .store mode p x,m => ((),writeWord m p (a.encode mode x))

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

/-- Ordinary local machine interface. Numeric steps expose only a single
scalar or one fixed ISA lane; there is no block/run/output simulation field. -/
structure LocalEventABI (α σ : Type) (a : Arithmetic α) where
  observe : σ → Memory α
  step : {β : Type} → Event α β → σ → β × σ
  stepLaw : ∀ (β : Type) (e : Event α β) (state : σ),
    ((step e state).1,observe (step e state).2)=e.eval a (observe state)

def Code.native (abi : LocalEventABI α σ a) (c : Code α β) (state : σ) : β × σ :=
  match c with
  | .pure x => (x,state)
  | .event e next => let (x,state') := abi.step e state; (next x).native abi state'

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

def gatherCode : {n : Nat} → (Fin n → Code α α) → Code α (Fin n → α)
  | 0,_ => pure Fin.elim0
  | n+1,f => do
      let x ← f 0
      let xs ← gatherCode (fun i : Fin n => f i.succ)
      pure (Fin.cases x xs)

 theorem eval_gather (a : Arithmetic α) (f : Fin n → Code α α) (value : Fin n → α)
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

def foldCode (f : Nat → α → Code α α) : Nat → α → Code α α
  | 0,z => pure z
  | n+1,z => do
      let v ← foldCode f n z
      f n v

 theorem eval_foldCode (a : Arithmetic α) (f : Nat → α → Code α α) (g : Nat → α → α)
    (m : Memory α) (h : ∀ i z,(f i z).eval a m=(g i z,m)) (n : Nat) (z : α) :
    (foldCode f n z).eval a m=(foldN g n z,m) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldCode,bind,eval_bind,ih,Prod.fst,Prod.snd,h,foldN]

def operandCode (mode : Mode) (K bk block laneOffset : Nat) (buffer row : Nat) : Code α α :=
  if K%bk=0 ∨ laneOffset<K-block*bk then emit (.read mode (buffer,row*max K 1+block*bk+laneOffset))
  else emit (.literal .zero)

@[simp] theorem eval_operandCode (a : Arithmetic α) (mode : Mode) (K bk block laneOffset : Nat)
    (buffer row : Nat) (m : Memory α) :
    (operandCode mode K bk block laneOffset buffer row).eval a m=
      (sourceOperand a mode K bk block laneOffset m buffer row,m) := by
  unfold operandCode sourceOperand
  split_ifs <;> rfl

def blockCode (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (m n block : Nat) (z : α) : Code α α :=
  let kind := selectedInstruction c mode
  foldCode (fun group seed => do
    let xs ← gatherCode (fun lane : Fin kind.width =>
      operandCode mode s.k (configs c).bk block (group*kind.width+lane.val) b.x m)
    let ys ← gatherCode (fun lane : Fin kind.width =>
      operandCode mode s.k (configs c).bk block (group*kind.width+lane.val) b.w n)
    emit (.instruction kind xs ys seed)) ((configs c).bk/kind.width) z

@[simp] theorem eval_blockCode (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (m n block : Nat) (z : α) (memory : Memory α) :
    (blockCode s b c mode m n block z).eval a memory=
      (sourceBlock a s b c mode memory m n block z,memory) := by
  simp only [blockCode,sourceBlock]
  apply eval_foldCode
  intro group seed
  simp only [bind,eval_bind,eval_gather a _ _ memory (fun i => eval_operandCode a _ _ _ _ _ _ _ _),Prod.fst,Prod.snd,eval_emit,Event.eval]

def accumulationCode (s : Shape) (b : Buffers) (c : Fin 98) (mode : Mode) (m n : Nat) : Code α α := do
  let zero ← emit (.literal .zero)
  let seed ← match b.bias with
    | none => pure zero
    | some p => do
        let bias ← emit (.read mode (p,n))
        emit (.add bias zero)
  foldCode (blockCode s b c mode m n) (s.blocks c) seed

@[simp] theorem eval_accumulationCode (a : Arithmetic α) (s : Shape) (b : Buffers) (c : Fin 98)
    (mode : Mode) (m n : Nat) (memory : Memory α) :
    (accumulationCode s b c mode m n).eval a memory=
      (sourceAccumulation a s b c mode memory m n,memory) := by
  cases hb : b.bias <;>
    simp [accumulationCode,sourceAccumulation,hb,bind,eval_bind,Event.eval,
      eval_foldCode a _ _ memory (fun block z => eval_blockCode a s b c mode m n block z memory),pure]

#axiomsClean Code.native_refines
#axiomsClean eval_gather
#axiomsClean eval_operandCode
#axiomsClean eval_blockCode
#axiomsClean eval_accumulationCode
end
end VeriTile.Bench.Repairs.TritonLinearActivation
