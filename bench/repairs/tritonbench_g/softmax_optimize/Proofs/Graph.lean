import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Recurrence

/-! Ordered primitive graphs for the actual FP32 source and local compiler
lowerings. In particular, an FMA node is never replaced by multiply then add.
Different output lanes may select different denominator trees. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Graph
open Recurrence

inductive Expr (ι : Type) where
  | input : ι → Expr ι
  | zero : Expr ι
  | negInf : Expr ι
  | maximum : Expr ι → Expr ι → Expr ι
  | subtract : Expr ι → Expr ι → Expr ι
  | exponential : Expr ι → Expr ι
  | multiply : Expr ι → Expr ι → Expr ι
  | add : Expr ι → Expr ι → Expr ι
  | fma : Expr ι → Expr ι → Expr ι → Expr ι
  | divide : Expr ι → Expr ι → Expr ι
  | ifNegInf : Expr ι → Expr ι → Expr ι → Expr ι
  deriving Repr

structure Operations (α : Type) extends Recurrence.Operations α where
  fusedMultiplyAdd : α → α → α → α
  divide : α → α → α

def eval {ι α : Type} (ops : Operations α) (input : ι → α) : Expr ι → α
  | .input i => input i
  | .zero => ops.zero
  | .negInf => ops.negInf
  | .maximum a b => ops.maximum (eval ops input a) (eval ops input b)
  | .subtract a b => ops.subtract (eval ops input a) (eval ops input b)
  | .exponential a => ops.exponential (eval ops input a)
  | .multiply a b => ops.multiply (eval ops input a) (eval ops input b)
  | .add a b => ops.add (eval ops input a) (eval ops input b)
  | .fma a b c => ops.fusedMultiplyAdd (eval ops input a) (eval ops input b) (eval ops input c)
  | .divide a b => ops.divide (eval ops input a) (eval ops input b)
  | .ifNegInf test yes no => if ops.isNegInf (eval ops input test)
      then eval ops input yes else eval ops input no

/-- Native code can reorder local operands. These choices record that order;
no commutativity law is demanded of opaque FP operations. -/
structure UpdateOrder where
  massMultiplyReversed : Bool
  massAddReversed : Bool
  deriving Repr

def sourceOrder : UpdateOrder := ⟨Bool.false, Bool.false⟩
/-- Observed in the pinned B≤128 profiles. Larger vector profiles can retain
the source addition order, so this is not a universal compiler-order law. -/
def observedLLVMOrder : UpdateOrder := ⟨Bool.true, Bool.true⟩
def observedVectorOrder : UpdateOrder := ⟨Bool.true, Bool.false⟩

def update {ι : Type} (order : UpdateOrder)
    (state : Expr ι × Expr ι) (input : Expr ι) : Expr ι × Expr ι :=
  let newMax := Expr.maximum state.1 input
  let weight := Expr.exponential (.subtract state.1 newMax)
  let product := if order.massMultiplyReversed then Expr.multiply state.2 weight
    else Expr.multiply weight state.2
  let oldMass := Expr.ifNegInf newMax state.2 product
  let newMass := Expr.ifNegInf input .zero (.exponential (.subtract input newMax))
  (newMax, if order.massAddReversed then .add newMass oldMass else .add oldMass newMass)

theorem eval_source_update {ι α : Type} (ops : Operations α) (input : ι → α)
    (state : Expr ι × Expr ι) (score : Expr ι) :
    (eval ops input (update sourceOrder state score).1,
      eval ops input (update sourceOrder state score).2) =
    Recurrence.step ops.toOperations
      (eval ops input state.1, eval ops input state.2) (eval ops input score) := rfl

/-- Each weighted leaf is a pair, allowing an exact fused edge into a sum.
The four fused constructors retain both product and accumulation operand order. -/
inductive SumPlan (ι : Type) where
  | zero : SumPlan ι
  | product : ι → Bool → SumPlan ι
  | add : SumPlan ι → SumPlan ι → SumPlan ι
  | fmaLeft : ι → Bool → SumPlan ι → SumPlan ι
  | fmaRight : SumPlan ι → ι → Bool → SumPlan ι
  deriving Repr

def SumPlan.leaves {ι : Type} : SumPlan ι → List ι
  | .zero => []
  | .product i _ => [i]
  | .add a b => a.leaves ++ b.leaves
  | .fmaLeft i _ rest => i :: rest.leaves
  | .fmaRight rest i _ => rest.leaves ++ [i]

def SumPlan.expression {ι κ : Type} (terms : ι → Expr κ × Expr κ) : SumPlan ι → Expr κ
  | .zero => .zero
  | .product i reverse => if reverse then .multiply (terms i).2 (terms i).1
      else .multiply (terms i).1 (terms i).2
  | .add a b => .add (a.expression terms) (b.expression terms)
  | .fmaLeft i reverse rest => if reverse
      then .fma (terms i).2 (terms i).1 (rest.expression terms)
      else .fma (terms i).1 (terms i).2 (rest.expression terms)
  | .fmaRight rest i reverse => if reverse
      then .fma (terms i).2 (terms i).1 (rest.expression terms)
      else .fma (terms i).1 (terms i).2 (rest.expression terms)

/-- Every output lane can carry its own reduction order. Its leaves must be a
permutation of all source lanes, with no invented or omitted weighted term. -/
structure DenominatorPlans (B : Nat) where
  plan : Fin B → SumPlan (Fin B)
  covers : ∀ outputLane, (plan outputLane).leaves.Perm (List.finRange B)

def SumPlan.split {ι : Type} : List ι → SumPlan ι
  | [] => .zero
  | i :: rest => .add (.product i Bool.false) (SumPlan.split rest)

theorem SumPlan.split_leaves {ι : Type} (items : List ι) :
    (SumPlan.split items).leaves = items := by
  induction items with
  | nil => rfl
  | cons i rest ih => simp [SumPlan.split, SumPlan.leaves, ih]

def splitPlans (B : Nat) : DenominatorPlans B where
  plan := fun _ => SumPlan.split (List.finRange B)
  covers := fun _ => by rw [SumPlan.split_leaves]

theorem denominator_plans_inhabited (B : Nat) : Nonempty (DenominatorPlans B) :=
  ⟨splitPlans B⟩

/-- Local intrinsic obligations only. These fields quantify primitive operands,
never a kernel result, execution, memory state, index, grid or output row. -/
structure LocalExact {α β : Type} (a : Operations α) (b : Operations β) (f : α → β) : Prop where
  zero : f a.zero = b.zero
  negInf : f a.negInf = b.negInf
  maximum : ∀ x y, f (a.maximum x y) = b.maximum (f x) (f y)
  subtract : ∀ x y, f (a.subtract x y) = b.subtract (f x) (f y)
  exponential : ∀ x, f (a.exponential x) = b.exponential (f x)
  multiply : ∀ x y, f (a.multiply x y) = b.multiply (f x) (f y)
  add : ∀ x y, f (a.add x y) = b.add (f x) (f y)
  fusedMultiplyAdd : ∀ x y z, f (a.fusedMultiplyAdd x y z) = b.fusedMultiplyAdd (f x) (f y) (f z)
  divide : ∀ x y, f (a.divide x y) = b.divide (f x) (f y)
  isNegInf : ∀ x, a.isNegInf x = b.isNegInf (f x)

theorem eval_map {ι α β : Type} (a : Operations α) (b : Operations β) (f : α → β)
    (h : LocalExact a b f) (input : ι → α) (e : Expr ι) :
    f (eval a input e) = eval b (fun i => f (input i)) e := by
  induction e with
  | input i => rfl
  | zero => exact h.zero
  | negInf => exact h.negInf
  | maximum x y hx hy => simpa only [eval, h.maximum, hx, hy]
  | subtract x y hx hy => simpa only [eval, h.subtract, hx, hy]
  | exponential x hx => simpa only [eval, h.exponential, hx]
  | multiply x y hx hy => simpa only [eval, h.multiply, hx, hy]
  | add x y hx hy => simpa only [eval, h.add, hx, hy]
  | fma x y z hx hy hz => simpa only [eval, h.fusedMultiplyAdd, hx, hy, hz]
  | divide x y hx hy => simpa only [eval, h.divide, hx, hy]
  | ifNegInf test yes no ht hy hn =>
    simp only [eval, h.isNegInf, ht]
    split <;> assumption

theorem local_exact_identity {α : Type} (ops : Operations α) : LocalExact ops ops id := by
  constructor <;> intros <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Graph
