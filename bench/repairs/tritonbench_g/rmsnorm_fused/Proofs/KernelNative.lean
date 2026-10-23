import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Runtime
import bench.tritonbench_g.rms_norm_triton.Proofs.Observed

namespace VeriTile.Bench.Repairs.RmsnormFused.Ordered
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra graphSum)
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

abbrev Word := BitVec 32

/-- Local IEEE/target instruction meanings, including FTZ and NaN payload
behavior. These and the binary64→FP32 ABI conversion are model boundaries. -/
structure Instructions where
  literal : ℝ → Word
  add : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  sqrt : Word → Word
  cast : Word → Word
  fma : Word → Word → Word → Word
  scalarABI : ℝ → Word

inductive Fusion where
  | split | left | right
  deriving DecidableEq
inductive Accumulator where
  | keep | eliminate | fuse
  deriving DecidableEq

/-- Compiler choices are syntax/position data and cannot inspect input words. -/
structure Policy where
  accumulator : Accumulator
  accumulatorSwap : Bool
  divideOne : Bool
  castIdentity : Bool
  epsilonFusion : Fusion
  epsilonSwap : Bool
  productSwap : Bool → Bool

structure Value where
  word : Word
  product : Option (Word × Word) := none
  constant : Option ℝ := none
  reciprocal : Bool := Bool.false

def atom (word : Word) : Value := ⟨word,none,none,Bool.false⟩
def literal (I : Instructions) (x : ℝ) : Value := ⟨I.literal x,none,some x,Bool.false⟩

def multiply (I : Instructions) (P : Policy) (x y : Value) : Value :=
  let a := if P.productSwap y.reciprocal then y.word else x.word
  let b := if P.productSwap y.reciprocal then x.word else y.word
  ⟨I.mul a b,some (a,b),none,Bool.false⟩

def add (I : Instructions) (fusion : Fusion) (swap : Bool) (x y : Value) : Value :=
  atom (match fusion,x.product,y.product with
    | .left,some (a,b),_ => I.fma a b y.word
    | .right,_,some (a,b) => I.fma a b x.word
    | _,_,_ => if swap then I.add y.word x.word else I.add x.word y.word)

def accumulate (I : Instructions) (P : Policy) (x : Value) : Value :=
  match P.accumulator with
  | .eliminate => x
  | .keep => add I .split P.accumulatorSwap (literal I 0) x
  | .fuse => add I .right P.accumulatorSwap (literal I 0) x

def divide (I : Instructions) (P : Policy) (x y : Value) : Value :=
  if P.divideOne && decide (y.constant=some 1) then x
  else ⟨I.div x.word y.word,none,none,decide (x.constant=some 1)⟩

def convert (I : Instructions) (P : Policy) (x : Value) : Value :=
  if P.castIdentity then x else atom (I.cast x.word)

/-- Ordered native reduction instructions, with explicit local FMA choices.
No sum callback or whole-row numerical premise is permitted. -/
inductive Tree (B : Nat) where
  | zero
  | lane (i : Fin B)
  | join (fusion : Fusion) (swap : Bool) (left right : Tree B)

def Tree.leaves : Tree B → List (Fin B)
  | .zero => []
  | .lane i => [i]
  | .join _ _ a b => a.leaves++b.leaves

structure Reductions where
  tree : (B lane : Nat) → Tree B
  complete : ∀ B lane,(tree B lane).leaves.Perm (List.finRange B)

def Tree.evaluate (I : Instructions) (xs : Fin B → Value) : Tree B → Value
  | .zero => literal I 0
  | .lane i => xs i
  | .join fusion swap a b => add I fusion swap (a.evaluate I xs) (b.evaluate I xs)

/-- Lower the graph emitted by the actual DSL. Runtime epsilon is an input
leaf, never identified by its Real annotation or confused with 0/1/N. -/
def lower (I : Instructions) (P : Policy) (R : Reductions) (lane : Nat) : Term Word → Value
  | .input value => atom value
  | .literal (some .fp32) .real value => literal I value
  | .cast (some .fp32) .real .real value => convert I P (lower I P R lane value)
  | .unary (some .fp32) .sqrt value => atom (I.sqrt (lower I P R lane value).word)
  | .binary (some .fp32) .real .mul a b => multiply I P (lower I P R lane a) (lower I P R lane b)
  | .binary (some .fp32) .real .div a b => divide I P (lower I P R lane a) (lower I P R lane b)
  | .binary (some .fp32) .real .add (.literal (some .fp32) .real z) b =>
      if z=0 then accumulate I P (lower I P R lane b)
      else add I P.epsilonFusion P.epsilonSwap (literal I z) (lower I P R lane b)
  | .binary (some .fp32) .real .add a b =>
      add I P.epsilonFusion P.epsilonSwap (lower I P R lane a) (lower I P R lane b)
  | .sum (some .fp32) B values => (R.tree B lane).evaluate I (fun i => lower I P R lane (values i))
  | _ => atom 0

def interpret (I : Instructions) (P : Policy) (R : Reductions) (lane : Nat) (term : Term Word) : Word :=
  (lower I P R lane term).word

/-- Independent native recurrence: masked lanes, zero accumulator handling,
ordered reduction, /N, runtime epsilon, sqrt, reciprocal, x scale, weight scale.
N=1 may retain the product through singleton reduction and /1 elimination,
allowing the compiler's x*x+epsilon FMA without equating it to split arithmetic. -/
def reference (I : Instructions) (P : Policy) (R : Reductions) (N B : Nat)
    (epsilon : Word) (xs ws : Nat → Word) (lane : Nat) : Word :=
  let inputs : Fin B → Value := fun j =>
    if j.val<N then convert I P (atom (xs j.val)) else literal I 0
  let squares := fun j => accumulate I P (multiply I P (inputs j) (inputs j))
  let total := (R.tree B lane).evaluate I squares
  let variance := divide I P total (literal I N)
  let radicand := add I P.epsilonFusion P.epsilonSwap variance (atom epsilon)
  let root := atom (I.sqrt radicand.word)
  let rstd := divide I P (literal I 1) root
  (multiply I P (multiply I P (convert I P (atom (xs lane))) rstd) (atom (ws lane))).word

def singletonPolicy : Policy :=
  ⟨.eliminate,Bool.false,Bool.true,Bool.true,.left,Bool.true,id⟩

/-- A concrete nontrivial lowering branch: singleton reduction and /1
elimination expose the product to the runtime-epsilon FMA. -/
theorem singleton_fma (I : Instructions) (R : Reductions) (epsilon : Word)
    (xs ws : Nat → Word) (tree : R.tree 1 0=.lane ⟨0,by decide⟩) :
    reference I singletonPolicy R 1 1 epsilon xs ws 0=
      I.mul (I.mul (I.div (I.literal 1) (I.sqrt (I.fma (xs 0) (xs 0) epsilon))) (xs 0)) (ws 0) := by
  simp [reference,tree,Tree.evaluate,singletonPolicy,convert,atom,accumulate,
    multiply,divide,literal,add]

theorem source_lowering (I : Instructions) (P : Policy) (R : Reductions)
    (memory : RegionName → Nat → Word) (N B row lane : Nat) (epsilon : Word)
    (s : State (Term Word)) (hx : ∀ i,Runtime.read s "X" i=.input (memory "X" i))
    (hw : ∀ i,Runtime.read s "W" i=.input (memory "W" i)) (active : lane<N) :
    interpret I P R lane (Runtime.value graphAlgebra s "X" "W" row N B (.input epsilon) lane)=
      reference I P R N B epsilon (fun i => memory "X" (row*N+i)) (memory "W") lane := by
  simp only [Runtime.value,Runtime.rstd,Runtime.variance,Runtime.square,Runtime.xload,Runtime.wload,
    Runtime.lit,Runtime.add,Runtime.mul,Runtime.div,Runtime.cast,graphAlgebra,graphSum,
    interpret,lower,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    reference,hx,hw,active,if_true,id_eq]
  norm_num only
  have squares : (fun j : Fin B =>
      accumulate I P (multiply I P
        (lower I P R lane (if j.val<N then
          Term.cast (some .fp32) .real .real (if j.val<N then .input (memory "X" (row*N+j.val))
            else .literal (some .fp32) .real 0) else .literal (some .fp32) .real 0))
        (lower I P R lane (if j.val<N then
          Term.cast (some .fp32) .real .real (if j.val<N then .input (memory "X" (row*N+j.val))
            else .literal (some .fp32) .real 0) else .literal (some .fp32) .real 0)))) =
      (fun j : Fin B => accumulate I P (multiply I P
        (if j.val<N then convert I P (atom (memory "X" (row*N+j.val))) else literal I 0)
        (if j.val<N then convert I P (atom (memory "X" (row*N+j.val))) else literal I 0))) := by
    funext j
    split_ifs <;> rfl
  rw [squares]

#guard_msgs (drop info) in
#specNonCircular reference avoiding [lower,interpret,Runtime.value,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Ordered
