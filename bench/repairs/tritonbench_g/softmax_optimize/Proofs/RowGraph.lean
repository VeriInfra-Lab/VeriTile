import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Graph

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.RowGraph
open Graph

def score (N B tileIndex lane : Nat) : Expr Nat :=
  if tileIndex * B + lane < N then .input (tileIndex * B + lane) else .negInf

def laneAfter (order : UpdateOrder) (N B lane count : Nat) : Expr Nat × Expr Nat :=
  (List.range count).foldl (fun state i => update order state (score N B i lane)) (.negInf, .zero)

def twoLoops (order : UpdateOrder) (N B lane : Nat) : Expr Nat × Expr Nat :=
  update order (laneAfter order N B lane (Config.fullTiles N B))
    (score N B (Config.fullTiles N B) lane)

theorem two_loops_complete (order : UpdateOrder) (N B lane : Nat) :
    twoLoops order N B lane = laneAfter order N B lane (Config.fullTiles N B + 1) := by
  simp [twoLoops, laneAfter, List.range_succ, List.foldl_append]

theorem eval_score {α : Type} (ops : Graph.Operations α) (input : Nat → α)
    (N B i lane : Nat) :
    eval ops input (score N B i lane) = Recurrence.laneInput ops.toOperations input N B i lane := by
  unfold score Recurrence.laneInput
  split <;> rfl

/-- Constructive correspondence between all guarded source updates and their
expression graph, for arbitrary total primitive operations and all inputs. -/
theorem eval_laneAfter {α : Type} (ops : Graph.Operations α) (input : Nat → α)
    (N B lane count : Nat) :
    (eval ops input (laneAfter sourceOrder N B lane count).1,
      eval ops input (laneAfter sourceOrder N B lane count).2) =
      Recurrence.laneAfter ops.toOperations input N B lane count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [laneAfter, Recurrence.laneAfter, List.range_succ, List.foldl_append,
      List.foldl_cons, List.foldl_nil] at ih ⊢
    rw [eval_source_update, eval_score]
    rw [ih]

theorem eval_twoLoops {α : Type} (ops : Graph.Operations α) (input : Nat → α)
    (N B lane : Nat) :
    (eval ops input (twoLoops sourceOrder N B lane).1,
      eval ops input (twoLoops sourceOrder N B lane).2) =
      Recurrence.twoLoops ops.toOperations input N B lane := by
  rw [two_loops_complete, Recurrence.two_loops_complete, eval_laneAfter]

inductive MaxPlan (ι : Type) where
  | input : ι → MaxPlan ι
  | node : MaxPlan ι → MaxPlan ι → MaxPlan ι
  deriving Repr

def MaxPlan.leaves {ι : Type} : MaxPlan ι → List ι
  | .input i => [i]
  | .node a b => a.leaves ++ b.leaves

def MaxPlan.expression {ι κ : Type} (terms : ι → Expr κ) : MaxPlan ι → Expr κ
  | .input i => terms i
  | .node a b => .maximum (a.expression terms) (b.expression terms)

/-- Replicated maximum values may retain different operand orders as well.
This matters for an all-word model with signed zeros and NaN payloads. -/
structure Profile (B : Nat) where
  updateOrder : UpdateOrder
  maxPlan : Fin B → MaxPlan (Fin B)
  maxCovers : ∀ lane, (maxPlan lane).leaves.Perm (List.finRange B)
  denominators : DenominatorPlans B

def maxima (profile : Profile B) (N : Nat) (lane : Fin B) : Expr Nat :=
  (profile.maxPlan lane).expression (fun i : Fin B => (twoLoops profile.updateOrder N B i).1)

/-- Terms are computed by their owning native lanes before the sum shuffle.
Each term uses that lane's maximum reduction result and exact product operands. -/
def factors (profile : Profile B) (N : Nat) (lane : Fin B) : Expr Nat × Expr Nat :=
  (.exponential (.subtract (twoLoops profile.updateOrder N B lane).1 (maxima profile N lane)),
    (twoLoops profile.updateOrder N B lane).2)

def denominator (profile : Profile B) (N : Nat) (lane : Fin B) : Expr Nat :=
  (profile.denominators.plan lane).expression (factors profile N)

def output (profile : Profile B) (N : Nat) (column : Nat) (lane : Fin B) : Expr Nat :=
  .divide (.exponential (.subtract (.input column) (maxima profile N lane)))
    (denominator profile N lane)

/-- Only local primitive conformance is required for graph interpretation;
the row's full symbolic expression is derived from those local equations. -/
theorem output_transfer {α β : Type} (a : Graph.Operations α) (b : Graph.Operations β)
    (f : α → β) (h : LocalExact a b f) (input : Nat → α)
    (profile : Profile B) (N column : Nat) (lane : Fin B) :
    f (eval a input (output profile N column lane)) =
      eval b (fun j => f (input j)) (output profile N column lane) :=
  eval_map a b f h input _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.RowGraph
