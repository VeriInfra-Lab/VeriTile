import bench.repairs.tritonbench_g.cross_entropy2.Proofs.TorchSplit

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
noncomputable section
set_option maxHeartbeats 8000000

def appendLeaves : Tree → List Nat → Tree
  | tree,[] => tree
  | tree,i::rest => appendLeaves (.node tree (.leaf i)) rest

theorem append_leaves (tree : Tree) (indices : List Nat) :
    (appendLeaves tree indices).leaves=tree.leaves++indices := by
  induction indices generalizing tree with
  | nil => simp [appendLeaves]
  | cons i rest ih => simp [appendLeaves,ih,Tree.leaves,List.append_assoc]

theorem append_run {α : Type} (op : α → α → α) (seed : α) (input : Nat → α)
    (tree : Tree) (indices : List Nat) :
    (appendLeaves tree indices).run op seed input=
      indices.foldl (fun acc i => op acc (input i)) (tree.run op seed input) := by
  induction indices generalizing tree with
  | nil => rfl
  | cons i rest ih => exact ih (.node tree (.leaf i))

def accumulatorIndices (count slot : Nat) : List Nat :=
  (List.range count).filter (fun i => i%4==slot)

private theorem count_filter_eq (p : Nat → Bool) (xs : List Nat) (i : Nat) :
    (xs.filter p).count i=if p i then xs.count i else 0 := by
  cases hp : p i with
  | «true» => simp [List.count_filter hp]
  | «false» =>
    simp only [Bool.false_eq_true,↓reduceIte]
    apply List.count_eq_zero.mpr
    simp [List.mem_filter,hp]

/-- The four residue classes partition all input leaves exactly once. The
tree retains each accumulator's explicit identity seed and combine order. -/
theorem accumulator_indices_cover (count : Nat) :
    (accumulatorIndices count 0++accumulatorIndices count 1++
      accumulatorIndices count 2++accumulatorIndices count 3).Perm (List.range count) := by
  apply List.perm_iff_count.mpr
  intro i
  have residue : i%4=0 ∨ i%4=1 ∨ i%4=2 ∨ i%4=3 := by omega
  rcases residue with h|h|h|h <;>
    simp only [accumulatorIndices,List.count_append,count_filter_eq,h] <;> simp

def threadFourTree (count : Nat) : Tree :=
  let accumulator := fun slot => appendLeaves .seed (accumulatorIndices count slot)
  .node (.node (.node (accumulator 0) (accumulator 1)) (accumulator 2)) (accumulator 3)

def threadFourPlan (count : Nat) (positive : 0<count) : Plan count where
  tree := threadFourTree count
  positive := positive
  covers := by
    simpa only [threadFourTree,Tree.leaves,append_leaves,List.nil_append] using accumulator_indices_cover count

/-- Exact arbitrary-length single-thread vt0=4 loop, including its tail.
The C++ loop distributes successive strided inputs cyclically over the four
accumulators, then combines accumulator 0 with 1, 2, and 3 in that order. -/
def threadFourProgram {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) (count : Nat) : α :=
  let accumulator := fun slot => (accumulatorIndices count slot).foldl (fun acc i => op acc (input i)) seed
  op (op (op (accumulator 0) (accumulator 1)) (accumulator 2)) (accumulator 3)

theorem thread_four_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) (count : Nat) :
    (threadFourTree count).run op seed input=threadFourProgram op seed input count := by
  simp only [threadFourTree,Tree.run,append_run,threadFourProgram]

theorem thread_two_specialization : threadFourTree 2=threadTwoFour.tree := rfl

#axiomsClean append_leaves
#axiomsClean append_run
#axiomsClean accumulator_indices_cover
#axiomsClean thread_four_exact
#axiomsClean thread_two_specialization
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
