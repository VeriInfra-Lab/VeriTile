import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardGrid

/-! The ordinary Torch dim0 reduction interface is parameterized by its actual
FP32 scalar addition schedule. Leaves load the derived [batch,dim] temporary,
and stores target the actual newly returned [dim] view. A tree is not a claim
that arbitrary schedules are numerically equal. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Reduction
open Kernels ByteMemory Execution

inductive Tree where
  | zero
  | leaf (batch : Nat)
  | add (left right : Tree)
  deriving DecidableEq, Repr

def Tree.leaves : Tree → List Nat
  | .zero => []
  | .leaf b => [b]
  | .add left right => left.leaves ++ right.leaves

def Tree.evaluate {α : Type} (zero : α) (add : α → α → α) (values : Nat → α) : Tree → α
  | .zero => zero
  | .leaf b => values b
  | .add left right => add (left.evaluate zero add values) (right.evaluate zero add values)

theorem Tree.evaluate_congr {α : Type} (zero : α) (add : α → α → α)
    (left right : Nat → α) (tree : Tree) (h : ∀ b ∈ tree.leaves, left b = right b) :
    tree.evaluate zero add left = tree.evaluate zero add right := by
  induction tree with
  | zero => rfl
  | leaf b => exact h b (by simp [Tree.leaves])
  | add l r ihl ihr =>
    simp only [Tree.evaluate]
    rw [ihl (by intro b hb; exact h b (by simp [Tree.leaves, hb])),
      ihr (by intro b hb; exact h b (by simp [Tree.leaves, hb]))]

structure Plan (batch dim components : Nat) where
  tree : Nat → Tree
  leaves : ∀ q, q < components * dim → (tree q).leaves.Perm (List.range batch)
  empty : batch = 0 → ∀ q, q < components * dim → tree q = .zero

theorem Plan.leaf_bound {batch dim components : Nat} (plan : Plan batch dim components)
    (q : Nat) (hq : q < components * dim) (b : Nat) (hb : b ∈ (plan.tree q).leaves) : b < batch := by
  exact List.mem_range.mp ((plan.leaves q hq).mem_iff.mp hb)

def inputWord (route : Route) (dim q b : Nat) := route.components * (b * dim) + q

def value {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (batch dim : Nat) (plan : Plan batch dim route.components) (input : View) (m : Heap) (q : Nat) : α :=
  (plan.tree q).evaluate ops.zero ops.add
    (fun b => load codec m (input.address (inputWord route dim q b)))

def execute {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (batch dim : Nat) (plan : Plan batch dim route.components) (input returned : View) (m : Heap) : Heap :=
  writeValues codec returned (List.range (route.components * dim))
    (value codec route ops batch dim plan input m) m

theorem returned_readback {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (batch dim : Nat) (plan : Plan batch dim route.components) (input returned : View) (m : Heap)
    (q : Nat) (hq : q < route.components * dim) :
    load codec (execute codec route ops batch dim plan input returned m) (returned.address q) =
      value codec route ops batch dim plan input m q := by
  exact writeValues_read codec returned _ _ m q (List.mem_range.mpr hq)

theorem inputWord_bounded (route : Route) (batch dim q b : Nat)
    (hq : q < route.components * dim) (hb : b < batch) :
    inputWord route dim q b < route.components * (batch * dim) := by
  have hc := components_pos route
  unfold inputWord
  nlinarith

/-- This bridge is applied only after backward-grid readback has derived each
batch contribution. The reduction interface itself mentions no SSM target. -/
theorem returned_from_contributions {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (batch dim : Nat) (plan : Plan batch dim route.components) (input returned : View) (m : Heap)
    (contributions : Nat → Nat → α)
    (actualInput : ∀ q, q < route.components * dim → ∀ b, b < batch →
      load codec m (input.address (inputWord route dim q b)) = contributions q b)
    (q : Nat) (hq : q < route.components * dim) :
    load codec (execute codec route ops batch dim plan input returned m) (returned.address q) =
      (plan.tree q).evaluate ops.zero ops.add (contributions q) := by
  rw [returned_readback codec route ops batch dim plan input returned m q hq]
  apply Tree.evaluate_congr
  intro b hb
  exact actualInput q hq b (plan.leaf_bound q hq b hb)

theorem empty_batch {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (dim : Nat) (plan : Plan 0 dim route.components) (input returned : View) (m : Heap)
    (q : Nat) (hq : q < route.components * dim) :
    load codec (execute codec route ops 0 dim plan input returned m) (returned.address q) = ops.zero := by
  rw [returned_readback codec route ops 0 dim plan input returned m q hq]
  simp [value, plan.empty rfl q hq, Tree.evaluate]

theorem full_frame {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (batch dim : Nat) (plan : Plan batch dim route.components) (input returned : View) (m : Heap)
    (size : returned.words = route.components * dim) (p : Nat)
    (outside : p < returned.pointer ∨ returned.pointer + 4 * returned.words ≤ p) :
    execute codec route ops batch dim plan input returned m p = m p := by
  apply writeValues_frame
  intro i hi
  have hi := List.mem_range.mp hi
  simp only [View.address]
  omega

theorem caller_frame {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (batch dim : Nat) (plan : Plan batch dim route.components) (input returned : View) (m : Heap)
    (size : returned.words = route.components * dim) (caller : Backing)
    (fresh : returned.backing.disjoint caller) :
    Frame m (execute codec route ops batch dim plan input returned m) caller := by
  apply writeValues_other_backing codec returned _ _ m caller fresh
  intro i hi
  rw [size]
  exact List.mem_range.mp hi

theorem Tree.real_value (tree : Tree) (values : Nat → ℂ) :
    tree.evaluate 0 (· + ·) values = (tree.leaves.map values).sum := by
  induction tree with
  | zero => rfl
  | leaf b => simp [Tree.evaluate, Tree.leaves]
  | add l r ihl ihr => simp [Tree.evaluate, Tree.leaves, ihl, ihr]

theorem plan_real_sum {batch dim components : Nat} (plan : Plan batch dim components)
    (q : Nat) (hq : q < components * dim) (values : Nat → ℂ) :
    (plan.tree q).evaluate 0 (· + ·) values = (Finset.range batch).sum values := by
  rw [Tree.real_value, (List.Perm.map values (plan.leaves q hq)).sum_eq]
  have sumRange (n : Nat) : ((List.range n).map values).sum = (Finset.range n).sum values := by
    induction n with
    | zero => simp
    | succ n ih => simp [List.range_succ, Finset.sum_range_succ, ih]
  exact sumRange batch

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Reduction
