import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.Execution
import VeriTile.Triton.Float.ProgramIO

/-! A task-local carrier for the already selected fixed-K8 accumulation.
The unchanged source executes through Structural. A dot records its ordered
eight-operand groups, and its original accumulator addition consumes those
groups with the actual live carry. This is the existing instruction profile,
not a numerical contraction law between separately rounded operations. -/
namespace VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

abbrev Group (α : Type) := (Fin 8 → α) × (Fin 8 → α)

inductive Node (α : Type) where
  | value (value : α)
  | dot (groups : List (Group α))

instance [Inhabited α] : Inhabited (Node α) := ⟨.value default⟩

def observe (p : Ops α) : Node α → α
  | .value x => x
  | .dot groups => groups.foldl (fun carry g => p.dot8 g.1 g.2 carry) p.zero

def consume (p : Ops α) (carry : Node α) (groups : List (Group α)) : Node α :=
  groups.foldl (fun old g => .value (p.dot8 g.1 g.2 (observe p old))) carry

/-- Scalar values retain exactly the old primitive operations. The sole
pending constructor is consumed at the source's dot/add boundary. -/
def ops (p : Ops α) : Ops (Node α) where
  zero := .value p.zero
  fromInt := fun z => .value (p.fromInt z)
  mul := fun a b => .value (p.mul (observe p a) (observe p b))
  atomicAdd := fun a b => match b with
    | .value b => .value (p.atomicAdd (observe p a) b)
    | .dot groups => consume p a groups
  dot8 := fun a b carry =>
    .value (p.dot8 (fun i => observe p (a i)) (fun i => observe p (b i)) (observe p carry))

def groups (p : Ops α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (i : TileIndex [m,n]) : List (Group α) :=
  (List.range (k/8)).map (fun chunk =>
    ((fun lane => observe p (if h : chunk*8+lane.val<k then
       a (i.1,⟨chunk*8+lane.val,h⟩,PUnit.unit) else (ops p).zero)),
     (fun lane => observe p (if h : chunk*8+lane.val<k then
       b (⟨chunk*8+lane.val,h⟩,i.2.1,PUnit.unit) else (ops p).zero))))

def product (p : Ops α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (i : TileIndex [m,n]) : Node α := .dot (groups p a b i)

def base (p : Ops α) (rest : Algebra α) : Algebra (Node α) where
  literal := fun precision dtype x => .value (rest.literal precision dtype x)
  negInf := .value rest.negInf
  binary := fun precision dtype op a b =>
    .value (rest.binary precision dtype op (observe p a) (observe p b))
  unary := fun precision op x => .value (rest.unary precision op (observe p x))
  cast := fun precision src dst x => .value (rest.cast precision src dst (observe p x))
  fromNat := fun precision n => .value (rest.fromNat precision n)
  fromInt := fun precision n => .value (rest.fromInt precision n)
  fp32Bits := fun bits => .value (rest.fp32Bits bits)
  fp32Load := id
  reduceMax := fun precision {_} ax keep vals i =>
    .value (rest.reduceMax precision ax keep (fun j => observe p (vals j)) i)
  reduceSum := fun precision {_} ax keep vals i =>
    .value (rest.reduceSum precision ax keep (fun j => observe p (vals j)) i)
  dot := fun _ {batch} {_} {_} {_} a b => match batch with
    | [] => some (product p a b)
    | _ :: _ => none

def engine (p : Ops α) (rest : Algebra α) : Algebra (Node α) :=
  KernelExecution.algebra (ops p) (base p rest)

theorem consume_groups (p : Ops α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (old : Values (Node α) .real [m,n]) (i : TileIndex [m,n]) :
    consume p (old i) (groups p a b i) = KernelExecution.dotTile (ops p) a b old i := by
  simp only [consume,groups,List.foldl_map,KernelExecution.dotTile,ops]

theorem consume_product (p : Ops α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (old : Values (Node α) .real [m,n]) (i : TileIndex [m,n]) :
    (engine p rest).binary (some .fp32) .real .add (old i) (product p a b i) =
      KernelExecution.dotTile (ops p) a b old i :=
  consume_groups p a b old i

/-- The shared evaluator and the established source-local fused evaluator
agree exactly at the recognized source node, even for arbitrary live carry. -/
theorem accumulating_step (p : Ops α) (rest : Algebra α) [Inhabited α]
    {m k n : Nat} (left : Op .real [m,k]) (right : Op .real [k,n])
    (s : FP.Structural.State (Node α)) :
    FP.Structural.step (engine p rest)
      (.assign .real [m,n] "accumulator" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "accumulator")
          (.dot (batch := []) left right))))) s =
    KernelExecution.localStep (ops p) (base p rest)
      (.assign .real [m,n] "accumulator" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "accumulator")
          (.dot (batch := []) left right))))) s := by
  let E := engine p rest
  have refEval : evalOp E (some .fp32) (.ref .real [m,n] "accumulator") s=s.regs .real [m,n] "accumulator" := by
    rw [evalOp_unfold]
  have dotEval : evalOp E (some .fp32)
      (.dot (batch:=[]) (M:=m) (K:=k) (N:=n) left right) s=
      (evalOp E (some .fp32) left s).bind (fun a=>
        (evalOp E (some .fp32) right s).bind (fun b=>some (product p a b))) := by
    rw [evalOp_unfold]
    rfl
  rw [KernelExecution.localStep_dot,FP.Structural.step]
  change (evalOp E (some .fp32)
      (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "accumulator")
        (.dot (batch:=[]) (M:=m) (K:=k) (N:=n) left right)) s).bind
      (fun v=>some (s.setReg "accumulator" .real [m,n] v))=_
  apply congrArg (fun v : Option (Values (Node α) .real [m,n])=>
    v.bind (fun v=>some (s.setReg "accumulator" .real [m,n] v)))
  rw [evalOp_unfold]
  change ((evalOp E (some .fp32) (.ref .real [m,n] "accumulator") s).bind fun old=>
      (evalOp E (some .fp32) (.dot (batch:=[]) (M:=m) (K:=k) (N:=n)
        left right) s).bind fun rhs=>
        some (bop (numeric E (some .fp32) .add .real) (.consSame (.consSame .nil)) old rhs))=_
  rw [refEval]
  cases hc : s.regs .real [m,n] "accumulator" with
  | none=>
    change none=KernelExecution.accumulate (ops p) (base p rest) left right (.ref .real [m,n] "accumulator") s
    unfold KernelExecution.accumulate
    change none=(evalOp E (some .fp32) left s).bind (fun a=>
      (evalOp E (some .fp32) right s).bind (fun b=>
        (evalOp E (some .fp32) (.ref .real [m,n] "accumulator") s).bind
          (fun old=>some (KernelExecution.dotTile (ops p) a b old))))
    rw [refEval,hc]
    cases evalOp E (some .fp32) left s <;> cases evalOp E (some .fp32) right s <;> rfl
  | some old=>
    simp only [Option.bind_some]
    rw [dotEval]
    unfold KernelExecution.accumulate
    change _=(evalOp E (some .fp32) left s).bind (fun a=>
      (evalOp E (some .fp32) right s).bind (fun b=>
        (evalOp E (some .fp32) (.ref .real [m,n] "accumulator") s).bind
          (fun carry=>some (KernelExecution.dotTile (ops p) a b carry))))
    rw [refEval,hc]
    cases ha : evalOp E (some .fp32) left s <;>
      cases hb : evalOp E (some .fp32) right s
    all_goals simp only [Option.bind_none,Option.bind_some]
    all_goals try rfl
    rename_i a b
    exact congrArg some (funext fun i=>consume_product (rest:=rest) p a b old i)

#axiomsClean consume_groups
#axiomsClean accumulating_step
end
end VeriTile.Bench.Repairs.Int4Matmul.SourceGraph
