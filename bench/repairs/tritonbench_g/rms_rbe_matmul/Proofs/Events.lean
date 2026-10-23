import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.Symbolic

namespace VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

inductive Primitive (F : Type) where
  | literal (v : F)
  | zero | one
  | add (x y : F)
  | mul (x y : F)
  | mulFTZ (x y : F)
  | divide (x y : F)
  | rsqrt (x : F)
  | neg (x : F)
  | fma (x y z : F)
  | pow (x y : F)
  | cos (x : F)
  | sin (x : F)
  | ofInt (n : Int)
  | mma8 (x y : Fin 8 → F) (old : F)

def Primitive.eval (a : Ops F) : Primitive F → F
  | .literal v => v
  | .zero => a.zero
  | .one => a.one
  | .add x y => a.add x y
  | .mul x y => a.mul x y
  | .mulFTZ x y => a.mulFTZ x y
  | .divide x y => a.divide x y
  | .rsqrt x => a.rsqrt x
  | .neg x => a.neg x
  | .fma x y z => a.fma x y z
  | .pow x y => a.pow x y
  | .cos x => a.cos x
  | .sin x => a.sin x
  | .ofInt n => a.ofInt n
  | .mma8 x y old => a.mma8 x y old

inductive Event (F H : Type) : Type → Type where
  | scalar (p : Primitive F) : Event F H F
  | input (kind : InputKind) (p : Address) : Event F H F
  | narrow (v : F) : Event F H H
  | quantize (v : F) : Event F H Byte
  | store (p : Address) (v : Stored F H) : Event F H Unit
def Event.eval (a : StorageOps F H) : Event F H β → Memory F H → β × Memory F H
  | .scalar p,m => (p.eval a.toOps,m)
  | .input kind p,m => (inputEval a kind m p,m)
  | .narrow v,m => (a.narrow v,m)
  | .quantize v,m => (a.e5Encode v,m)
  | .store p v,m => ((),write m p v)
inductive Code (F H : Type) : Type → Type 1 where
  | pure (x : β) : Code F H β
  | event (e : Event F H β) (next : β → Code F H γ) : Code F H γ
def Code.bind (c : Code F H β) (f : β → Code F H γ) : Code F H γ :=
  match c with
  | .pure x => f x
  | .event e next => .event e (fun x => (next x).bind f)
instance : Monad (Code F H) where
  pure := Code.pure
  bind := Code.bind
def emit (e : Event F H β) : Code F H β := .event e .pure
def Code.eval (a : StorageOps F H) (c : Code F H β) (m : Memory F H) : β × Memory F H :=
  match c with
  | .pure x => (x,m)
  | .event e next => let (x,m') := e.eval a m; (next x).eval a m'
@[simp] theorem eval_pure (a : StorageOps F H) (x : β) (m : Memory F H) :
    (Code.pure x).eval a m=(x,m) := rfl
@[simp] theorem eval_emit (a : StorageOps F H) (e : Event F H β) (m : Memory F H) :
    (emit e).eval a m=e.eval a m := by cases e <;> rfl
@[simp] theorem eval_bind (a : StorageOps F H) (c : Code F H β) (f : β → Code F H γ) (m : Memory F H) :
    (c.bind f).eval a m=(f (c.eval a m).1).eval a (c.eval a m).2 := by
  induction c generalizing m with
  | pure x => rfl
  | event e next ih => simp only [Code.bind,Code.eval]; exact ih _ _ _
def vectorCode : (n : Nat) → (Fin n → Code F H β) → Code F H (Fin n → β)
  | 0,_ => pure Fin.elim0
  | n+1,f => do
      let x ← f 0
      let xs ← vectorCode n (fun i => f i.succ)
      pure (Fin.cases x xs)
theorem eval_vectorCode (a : StorageOps F H) (n : Nat) (f : Fin n → Code F H β)
    (v : Fin n → β) (m : Memory F H) (h : ∀ i,(f i).eval a m=(v i,m)) :
    (vectorCode n f).eval a m=(v,m) := by
  induction n with
  | zero => simp [vectorCode,pure]; congr; funext i; exact Fin.elim0 i
  | succ n ih =>
    simp only [vectorCode,bind,eval_bind,h,Prod.fst,Prod.snd,ih _ _ (fun i => h i.succ),pure,eval_pure]
    congr 1
    funext i
    exact Fin.cases rfl (fun _ => rfl) i

def Expr.code : Expr F → Code F H F
  | .literal v => emit (.scalar (.literal v))
  | .input kind p => emit (.input kind p)
  | .zero => emit (.scalar .zero)
  | .one => emit (.scalar .one)
  | .add x y => do
      let x ← x.code
      let y ← y.code
      emit (.scalar (.add x y))
  | .mul x y => do
      let x ← x.code
      let y ← y.code
      emit (.scalar (.mul x y))
  | .mulFTZ x y => do
      let x ← x.code
      let y ← y.code
      emit (.scalar (.mulFTZ x y))
  | .divide x y => do
      let x ← x.code
      let y ← y.code
      emit (.scalar (.divide x y))
  | .rsqrt x => do
      let x ← x.code
      emit (.scalar (.rsqrt x))
  | .neg x => do
      let x ← x.code
      emit (.scalar (.neg x))
  | .fma x y z => do
      let x ← x.code
      let y ← y.code
      let z ← z.code
      emit (.scalar (.fma x y z))
  | .pow x y => do
      let x ← x.code
      let y ← y.code
      emit (.scalar (.pow x y))
  | .cos x => do
      let x ← x.code
      emit (.scalar (.cos x))
  | .sin x => do
      let x ← x.code
      emit (.scalar (.sin x))
  | .ofInt n => emit (.scalar (.ofInt n))
  | .mma8 x y old => do
      let xs ← vectorCode 8 (fun i => (x i).code)
      let ys ← vectorCode 8 (fun i => (y i).code)
      let previous ← old.code
      emit (.scalar (.mma8 xs ys previous))
theorem Expr.code_eval (a : StorageOps F H) (m : Memory F H) (e : Expr F) :
    e.code.eval a m=(e.eval a m,m) := by
  induction e with
  | mma8 x y old ix iy io =>
    simp only [Expr.code,bind,eval_bind,eval_vectorCode a 8 _ _ m ix,
      eval_vectorCode a 8 _ _ m iy,Prod.fst,Prod.snd,io,eval_emit,Event.eval,Primitive.eval,Expr.eval]
  | _ => simp_all only [Expr.code,bind,eval_bind,eval_emit,Event.eval,Primitive.eval,Expr.eval,Prod.fst,Prod.snd]

def cellCode (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (x : Cell s) : Code F H Unit := do
  let v ← (cellExpr p b plans reductions x).code
  if b.useByte then
    let byte ← emit (.quantize v)
    emit (.store (outputAddress b x) (.inl byte))
  else
    let half ← emit (.narrow v)
    emit (.store (outputAddress b x) (.inr (.inr half)))
theorem cellCode_eval (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (x : Cell s) (m : Memory F H) :
    (cellCode p b plans reductions x).eval a m=
      ((),write m (outputAddress b x) (cellValue a p b plans reductions m x)) := by
  rw [←cellExpr_eval a p b plans reductions m x]
  unfold cellCode
  simp only [bind,eval_bind,Expr.code_eval,Prod.fst,Prod.snd]
  split_ifs <;> simp_all [encode,bind,eval_bind,eval_emit,Event.eval] <;> split_ifs <;> simp_all
def gridCode (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) : List (Cell s) → Code F H Unit
  | [] => pure ()
  | x::xs => do cellCode p b plans reductions x; gridCode p b plans reductions xs
theorem gridCode_eval (a : StorageOps F H) (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (events : List (Cell s)) (m : Memory F H) :
    (gridCode p b plans reductions events).eval a m=((),run a p b plans reductions events m) := by
  induction events generalizing m with
  | nil => rfl
  | cons x xs ih => simp only [gridCode,bind,eval_bind,cellCode_eval,Prod.fst,Prod.snd,ih,run]

/-- Ordinary local machine ABI. Each law is one scalar/fixed-K8 operation,
one current-memory scalar load/conversion, or one scalar store. No execution,
reduction, projection, output, successful-launch or frame oracle is exposed. -/
structure LocalEventABI (F H σ : Type) (a : StorageOps F H) where
  observe : σ → Memory F H
  step : {β : Type} → Event F H β → σ → β × σ
  stepLaw : ∀ (β : Type) (e : Event F H β) (state : σ),
    ((step e state).1,observe (step e state).2)=e.eval a (observe state)
def Code.native (abi : LocalEventABI F H σ a) (c : Code F H β) (state : σ) : β × σ :=
  match c with
  | .pure x => (x,state)
  | .event e next => let (x,state') := abi.step e state; (next x).native abi state'
theorem Code.native_refines (abi : LocalEventABI F H σ a) (c : Code F H β) (state : σ) :
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
theorem native_grid (a : StorageOps F H) (abi : LocalEventABI F H σ a)
    (p : Parameters F) (b : Buffers)
    (plans : Nat → Nat → Nat → Nat → DotPlan)
    (reductions : Nat → Nat → Reduction) (events : List (Cell s)) (state : σ) :
    abi.observe ((gridCode p b plans reductions events).native abi state).2=
      run a p b plans reductions events (abi.observe state) := by
  have h := congrArg Prod.snd (Code.native_refines abi (gridCode p b plans reductions events) state)
  simpa only [gridCode_eval,Prod.snd] using h

#axiomsClean Expr.code_eval
#axiomsClean cellCode_eval
#axiomsClean gridCode_eval
#axiomsClean Code.native_refines
#axiomsClean native_grid
end
end VeriTile.Bench.Repairs.RmsRbeMatmul
