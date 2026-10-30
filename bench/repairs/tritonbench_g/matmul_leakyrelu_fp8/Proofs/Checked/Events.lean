import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Real
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000
abbrev foldN := @TritonLinearActivation.foldN

/-- One named SM90 WGMMA K8 result lane. The local operands are exactly that
lane's two K8 fragments and supplied FP32 accumulator. -/
structure Operations (α : Type) where
  zero : α
  castHalf : α → α
  nonnegative : α → Bool
  slopeMultiply : α → α
  instruction : (Fin 8 → α) → (Fin 8 → α) → α → α

def activate (ops : Operations α) (enabled : Bool) (value : α) : α :=
  if enabled then if ops.nonnegative value then value else ops.slopeMultiply value else value

def block (ops : Operations α) (xs ys : Fin 64 → α) (seed : α) : α :=
  (List.finRange 8).foldl (fun z group => ops.instruction
    (fun lane => xs ⟨group.val*8+lane.val,by omega⟩)
    (fun lane => ys ⟨group.val*8+lane.val,by omega⟩) z) seed

/-- The only source width is BK64; it is explicitly eight fixed K8 operations.
The unused other-width branch introduces no numerical operation. -/
def sourceArithmetic (ops : Operations α) : Source.Arithmetic α α where
  inputZero := ops.zero
  accumulatorZero := ops.zero
  dot width xs ys acc := fun row column =>
    if width64 : width = 64 then
      block ops (fun k => xs row (Fin.cast width64.symm k))
        (fun k => ys (Fin.cast width64.symm k) column) (acc row column)
    else acc row column
  activation := activate ops
  cast _ := ops.castHalf

inductive Event (α : Type) : Type → Type where
  | read (p : Address) : Event α α
  | zero : Event α α
  | castHalf (x : α) : Event α α
  | nonnegative (x : α) : Event α Bool
  | slopeMultiply (x : α) : Event α α
  | instruction (xs ys : Fin 8 → α) (seed : α) : Event α α
  | storeHalf (p : Address) (x : α) : Event α Unit

def write (memory : Memory α) (p : Address) (x : α) : Memory α := fun q => if q = p then x else memory q

def Event.eval (ops : Operations α) : Event α β → Memory α → β × Memory α
  | .read p,memory => (memory p,memory)
  | .zero,memory => (ops.zero,memory)
  | .castHalf x,memory => (ops.castHalf x,memory)
  | .nonnegative x,memory => (ops.nonnegative x,memory)
  | .slopeMultiply x,memory => (ops.slopeMultiply x,memory)
  | .instruction xs ys z,memory => (ops.instruction xs ys z,memory)
  | .storeHalf p x,memory => ((),write memory p x)

inductive Code (α : Type) : Type → Type 1 where
  | pure (value : β) : Code α β
  | event (event : Event α β) (next : β → Code α γ) : Code α γ

def Code.bind (code : Code α β) (next : β → Code α γ) : Code α γ :=
  match code with
  | .pure x => next x
  | .event e rest => .event e (fun x => (rest x).bind next)
instance : Monad (Code α) where pure := Code.pure; bind := Code.bind

def emit (event : Event α β) : Code α β := .event event .pure

def Code.eval (ops : Operations α) (code : Code α β) (memory : Memory α) : β × Memory α :=
  match code with
  | .pure x => (x,memory)
  | .event e rest => let step := e.eval ops memory; (rest step.1).eval ops step.2

@[simp] theorem eval_pure (ops : Operations α) (x : β) (memory : Memory α) :
    (Code.pure x).eval ops memory = (x,memory) := rfl
@[simp] theorem eval_emit (ops : Operations α) (event : Event α β) (memory : Memory α) :
    (emit event).eval ops memory = event.eval ops memory := by cases event <;> rfl
@[simp] theorem eval_bind (ops : Operations α) (code : Code α β) (next : β → Code α γ) (memory : Memory α) :
    (code.bind next).eval ops memory = (next (code.eval ops memory).1).eval ops (code.eval ops memory).2 := by
  induction code generalizing memory with
  | pure x => rfl
  | event e rest ih => simp only [Code.bind,Code.eval]; exact ih _ _ _

/-- Ordinary one-event machine laws, with no whole-kernel completion/result
or numerical-reference condition. All program composition is proved below. -/
structure LocalABI (α σ : Type) (ops : Operations α) where
  observe : σ → Memory α
  step : {β : Type} → Event α β → σ → β × σ
  law : ∀ (β : Type) (event : Event α β) state,
    ((step event state).1,observe (step event state).2) = event.eval ops (observe state)

def Code.native (abi : LocalABI α σ ops) : Code α β → σ → β × σ
  | .pure x,state => (x,state)
  | .event e rest,state => let step := abi.step e state; (rest step.1).native abi step.2

theorem native_refines (abi : LocalABI α σ ops) (code : Code α β) (state : σ) :
    ((code.native abi state).1,abi.observe (code.native abi state).2) = code.eval ops (abi.observe state) := by
  induction code generalizing state with
  | pure x => rfl
  | event e rest ih =>
    have h := abi.law _ e state
    have hx := congrArg Prod.fst h
    have hm := congrArg Prod.snd h
    simp only [Prod.fst,Prod.snd] at hx hm
    simp only [Code.native,Code.eval]
    rw [ih,hm,hx]

def gather : {n : Nat} → (Fin n → Code α β) → Code α (Fin n → β)
  | 0,_ => pure Fin.elim0
  | n+1,f => do
    let first ← f 0
    let rest ← gather (fun i : Fin n => f i.succ)
    pure (Fin.cases first rest)

theorem eval_gather (ops : Operations α) (f : Fin n → Code α β) (value : Fin n → β)
    (memory : Memory α) (h : ∀ i, (f i).eval ops memory = (value i,memory)) :
    (gather f).eval ops memory = (value,memory) := by
  induction n with
  | zero => simp only [gather,pure,eval_pure]; congr 1; funext i; exact Fin.elim0 i
  | succ n ih =>
    simp only [gather,bind,eval_bind,h,Prod.fst,Prod.snd]
    rw [ih (fun i => f i.succ) (fun i => value i.succ) (fun i => h i.succ)]
    simp only [Prod.fst,Prod.snd,pure,eval_pure]
    congr 1
    funext i
    exact Fin.cases rfl (fun _ => rfl) i

def foldCode (step : β → α → Code α α) : List β → α → Code α α
  | [],z => pure z
  | i::rest,z => do let value ← step i z; foldCode step rest value

theorem eval_fold (ops : Operations α) (step : β → α → Code α α) (f : β → α → α)
    (memory : Memory α) (h : ∀ i z, (step i z).eval ops memory = (f i z,memory)) (items : List β) (z : α) :
    (foldCode step items z).eval ops memory = (items.foldl (fun z i => f i z) z,memory) := by
  induction items generalizing z with
  | nil => rfl
  | cons i rest ih => simp only [foldCode,bind,eval_bind,h,Prod.fst,Prod.snd,ih,List.foldl_cons]

#axiomsClean native_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
