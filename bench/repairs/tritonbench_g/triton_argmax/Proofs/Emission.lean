import bench.repairs.tritonbench_g.triton_argmax.Proofs.Tree

/-! Ordered reduction interpretation from a selected SSA schedule. Register
references are checked locally by their finite indices. A choose instruction
uses one predicate for both payload and index; no associativity or non-NaN law
is used. The selected artifact supplies syntax, never a reduction-result law. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Emission
open Reduction Source

/-- One register is appended by each load, transport, or paired select. Move
includes register copies and lane/shuffle/shared-memory transport after the
selected producer has been identified. Operands retain their emitted order. -/
inductive Plan (lanes : Nat) : Nat → Type where
  | result {registers} : Fin registers → Plan lanes registers
  | load {registers} : Fin lanes → Plan lanes (registers+1) → Plan lanes registers
  | move {registers} : Fin registers → Plan lanes (registers+1) → Plan lanes registers
  | choose {registers} : Fin registers → Fin registers → Plan lanes (registers+1) → Plan lanes registers

def append {α : Type} {n : Nat} (registers : Fin n → α) (value : α) : Fin (n+1) → α :=
  Fin.lastCases value registers

def Plan.tree {n r : Nat} : Plan n r → (Fin r → Tree n) → Tree n
  | .result i,registers => registers i
  | .load i rest,registers => rest.tree (append registers (.leaf i))
  | .move i rest,registers => rest.tree (append registers (registers i))
  | .choose i j rest,registers => rest.tree (append registers (.fork (registers i) (registers j)))

/-- The scalar operations are exactly the emitted ordered greater/equal,
index comparison, boolean operations, and paired selects. -/
def choosePair {α : Type} {n : Nat} (gt eq : α → α → Bool) (a b : α × Fin n) : α × Fin n :=
  let condition := gt a.1 b.1 || (eq a.1 b.1 && decide (a.2.val < b.2.val))
  (if condition then a.1 else b.1,if condition then a.2 else b.2)

def Plan.run {α : Type} {n r : Nat} (gt eq : α → α → Bool) (input : Fin n → α) :
    Plan n r → (Fin r → α × Fin n) → α × Fin n
  | .result i,registers => registers i
  | .load i rest,registers => rest.run gt eq input (append registers (input i,i))
  | .move i rest,registers => rest.run gt eq input (append registers (registers i))
  | .choose i j rest,registers => rest.run gt eq input (append registers (choosePair gt eq (registers i) (registers j)))

theorem select_pair {α : Type} {n : Nat} (input : Fin n → α) (condition : Bool) (i j : Fin n) :
    (if condition then input i else input j,if condition then i else j)=
      (input (if condition then i else j),if condition then i else j) := by
  cases condition <;> rfl

theorem choose_tree {α : Type} {n : Nat} (gt eq : α → α → Bool) (input : Fin n → α)
    (a b : Tree n) :
    choosePair gt eq (a.eval gt eq input) (b.eval gt eq input)=(Tree.fork a b).eval gt eq input := by
  exact select_pair input
    (leftWins gt eq (input (a.pick gt eq input)) (input (b.pick gt eq input))
      (a.pick gt eq input).val (b.pick gt eq input).val)
    (a.pick gt eq input) (b.pick gt eq input)

theorem append_map {α β : Type} {n : Nat} (f : α → β) (registers : Fin n → α) (value : α) :
    append (fun i => f (registers i)) (f value)=fun i => f (append registers value i) := by
  funext i
  refine Fin.lastCases ?_ (fun j => ?_) i <;> simp [append]

theorem run_tree {α : Type} {n r : Nat} (gt eq : α → α → Bool) (input : Fin n → α)
    (plan : Plan n r) (registers : Fin r → Tree n) :
    plan.run gt eq input (fun i => (registers i).eval gt eq input)=
      (plan.tree registers).eval gt eq input := by
  induction plan with
  | result i => rfl
  | load i rest ih =>
    dsimp only [Plan.run,Plan.tree]
    have paired : (input i,i)=(Tree.leaf i).eval gt eq input := rfl
    rw [paired,append_map]
    exact ih _
  | move i rest ih =>
    dsimp only [Plan.run,Plan.tree]
    rw [append_map]
    exact ih _
  | choose i j rest ih =>
    dsimp only [Plan.run,Plan.tree]
    rw [choose_tree,append_map]
    exact ih _

abbrev Closed (n : Nat) := Plan n 0

def Closed.toTree {n : Nat} (plan : Closed n) : Tree n := Plan.tree plan Fin.elim0

def Closed.execute {α : Type} {n : Nat} (plan : Closed n) (gt eq : α → α → Bool)
    (input : Fin n → α) : α × Fin n := Plan.run gt eq input plan Fin.elim0

theorem closed_run_tree {α : Type} {n : Nat} (plan : Closed n) (gt eq : α → α → Bool)
    (input : Fin n → α) : plan.execute gt eq input=plan.toTree.eval gt eq input := by
  have empty : (fun i : Fin 0 => (Fin.elim0 i : Tree n).eval gt eq input)=Fin.elim0 := by
    funext i
    exact Fin.elim0 i
  have result := run_tree gt eq input plan Fin.elim0
  rw [empty] at result
  exact result

/-- Coverage is a finite syntactic property of the interpreted producer graph.
It is checked on the selected artifact, independently of values/comparators. -/
structure Artifact (n : Nat) where
  tree : Tree n
  covers : tree.Complete

inductive Stage where
  | first | second | dimension
  deriving DecidableEq,Repr

/-- The selected key retains the actual device/compiler target, source hash,
argument types, constexpr branch, reduction extent, and launch options. -/
structure Key where
  stage : Stage
  sourceHash : String
  target : String
  argumentTypes : List String
  block : Nat
  index64 : Bool
  warps : Nat
  options : List (String × String)
  deriving DecidableEq,Repr

#axiomsClean run_tree
#axiomsClean closed_run_tree
end VeriTile.Bench.Repairs.TritonArgmax.Emission
