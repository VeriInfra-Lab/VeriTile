import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Execution
import VeriTile.Triton.Float.ProgramIO

/-! Task-local graph carrier for the already selected dot/add contraction.
The actual source still executes through Structural. A dot node records its
ordered operand pairs; the source accumulator addition consumes that node with
its actual live carry. This is the fixed compiler-profile interpretation, not
a numerical equality between unfused addition and fused multiplication.
No source AST, shared executor or callback selecting a final answer is added. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

inductive Node (α : Type) where
  | value (value : α)
  | product (operands : List (α × α))

instance [Inhabited α] : Inhabited (Node α) := ⟨.value default⟩

def observe (p : Scalar α) : Node α → α
  | .value x => x
  | .product xs => xs.foldl (fun carry ab=>p.fma ab.1 ab.2 carry) p.zero

def scalar (p : Scalar α) : Scalar (Node α) where
  zero := .value p.zero
  fma := fun a b carry=>.value (p.fma (observe p a) (observe p b) (observe p carry))

def consume (p : Scalar α) (carry : Node α) (xs : List (α × α)) : Node α :=
  xs.foldl (fun old ab=>.value (p.fma ab.1 ab.2 (observe p old))) carry

/-- The retained dot node contains its exact ordered input pairs. -/
def product (p : Scalar α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (i : TileIndex [m,n]) : Node α :=
  .product ((List.range k).map fun j=>
    (observe p (if h : j<k then a (i.1,⟨j,h⟩,PUnit.unit) else (scalar p).zero),
     observe p (if h : j<k then b (⟨j,h⟩,i.2.1,PUnit.unit) else (scalar p).zero)))

def base (p : Scalar α) (rest : Algebra α) : Algebra (Node α) where
  literal := fun precision dtype x=>.value (rest.literal precision dtype x)
  negInf := .value rest.negInf
  binary := fun precision dtype op a b=>match op,b with
    | .add,.product xs=>consume p a xs
    | _,_=>.value (rest.binary precision dtype op (observe p a) (observe p b))
  unary := fun precision op x=>.value (rest.unary precision op (observe p x))
  cast := fun precision src dst x=>.value (rest.cast precision src dst (observe p x))
  fromNat := fun precision n=>.value (rest.fromNat precision n)
  fromInt := fun precision n=>.value (rest.fromInt precision n)
  fp32Bits := fun bits=>.value (rest.fp32Bits bits)
  fp32Load := id
  reduceMax := fun precision {_} ax keep vals i=>
    .value (rest.reduceMax precision ax keep (fun j=>observe p (vals j)) i)
  reduceSum := fun precision {_} ax keep vals i=>
    .value (rest.reduceSum precision ax keep (fun j=>observe p (vals j)) i)
  dotWithOptions := fun _ options {batch} {_} {_} {_} a b=>
    if options={allowTF32:=some Bool.false} then
      match batch with
      | []=>some (product p a b)
      | _::_=>none
    else none

def engine (p : Scalar α) (rest : Algebra α) : Algebra (Node α) :=
  Execution.algebra (scalar p) (base p rest)

theorem consume_fold (p : Scalar α) (a b : Nat → Node α) (count : Nat) (carry : Node α) :
    consume p carry ((List.range count).map (fun j=>(observe p (a j),observe p (b j))))=
      Execution.fmaFold (scalar p) a b count carry := by
  induction count with
  | zero=>rfl
  | succ count ih=>
    simp only [List.range_succ,List.map_append,List.map_cons,List.map_nil,
      consume,List.foldl_append,List.foldl_cons,List.foldl_nil,Execution.fmaFold] at *
    rw [ih]
    rfl

theorem consume_product (p : Scalar α) {m k n : Nat}
    (a : Values (Node α) .real [m,k]) (b : Values (Node α) .real [k,n])
    (old : Values (Node α) .real [m,n]) (i : TileIndex [m,n]) :
    (engine p rest).binary (some .fp32) .real .add (old i) (product p a b i)=
      Execution.dotTile (scalar p) a b old i := by
  exact consume_fold p _ _ k (old i)

/-- Exact agreement with the existing live-carry localStep at the original
recognized source node, for arbitrary input graphs and arbitrary live carry. -/
theorem accumulating_step (p : Scalar α) (rest : Algebra α) [Inhabited α]
    {m k n : Nat} (left : Op .real [m,k]) (right : Op .real [k,n])
    (s : State (Node α)) :
    FP.Structural.step (engine p rest)
      (.assign .real [m,n] "o" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "o")
          (.dot (batch:=[]) left right {allowTF32:=some Bool.false}))))) s=
    Execution.localStep (scalar p) (base p rest)
      (.assign .real [m,n] "o" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "o")
          (.dot (batch:=[]) left right {allowTF32:=some Bool.false}))))) s := by
  let E := engine p rest
  have refEval : evalOp E (some .fp32) (.ref .real [m,n] "o") s=s.regs .real [m,n] "o" := by
    rw [evalOp_unfold]
  have dotEval : evalOp E (some .fp32)
      (.dot (batch:=[]) (M:=m) (K:=k) (N:=n) left right {allowTF32:=some Bool.false}) s=
      (evalOp E (some .fp32) left s).bind (fun a=>
        (evalOp E (some .fp32) right s).bind (fun b=>some (product p a b))) := by
    rw [evalOp_unfold]
    rfl
  rw [Execution.localStep_dot,FP.Structural.step]
  change (evalOp E (some .fp32)
      (.add .real (.consSame (.consSame .nil)) (.ref .real [m,n] "o")
        (.dot (batch:=[]) (M:=m) (K:=k) (N:=n) left right {allowTF32:=some Bool.false})) s).bind
      (fun v=>some (s.setReg "o" .real [m,n] v))=_
  apply congrArg (fun v : Option (Values (Node α) .real [m,n])=>
    v.bind (fun v=>some (s.setReg "o" .real [m,n] v)))
  rw [evalOp_unfold]
  change ((evalOp E (some .fp32) (.ref .real [m,n] "o") s).bind fun old=>
      (evalOp E (some .fp32) (.dot (batch:=[]) (M:=m) (K:=k) (N:=n)
        left right {allowTF32:=some Bool.false}) s).bind fun rhs=>
        some (bop (numeric E (some .fp32) .add .real) (.consSame (.consSame .nil)) old rhs))=_
  rw [refEval]
  cases hc : s.regs .real [m,n] "o" with
  | none=>
    change none=Execution.accumulate (scalar p) (base p rest) left right (.ref .real [m,n] "o") s
    unfold Execution.accumulate
    change none=(evalOp E (some .fp32) left s).bind (fun a=>
      (evalOp E (some .fp32) right s).bind (fun b=>
        (evalOp E (some .fp32) (.ref .real [m,n] "o") s).bind
          (fun old=>some (Execution.dotTile (scalar p) a b old))))
    rw [refEval,hc]
    cases evalOp E (some .fp32) left s <;> cases evalOp E (some .fp32) right s <;> rfl
  | some old=>
    simp only [Option.bind_some]
    rw [dotEval]
    unfold Execution.accumulate
    change _=(evalOp E (some .fp32) left s).bind (fun a=>
      (evalOp E (some .fp32) right s).bind (fun b=>
        (evalOp E (some .fp32) (.ref .real [m,n] "o") s).bind
          (fun carry=>some (Execution.dotTile (scalar p) a b carry))))
    rw [refEval,hc]
    cases ha : evalOp E (some .fp32) left s <;>
      cases hb : evalOp E (some .fp32) right s
    all_goals simp only [Option.bind_none,Option.bind_some]
    all_goals try rfl
    rename_i a b
    exact congrArg some (funext fun i=>consume_product (rest:=rest) p a b old i)

#axiomsClean accumulating_step
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
