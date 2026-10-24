import Mathlib.Tactic.IntervalCases
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.TorchPartition

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 100000

/-- CUDA's descending offsets, shared-memory stages first when necessary,
then shuffle-down stages. Only lane zero is subsequently stored. -/
def downOffsets (power : Nat) : List Nat := (List.range power).reverse.map (fun i => 2^i)
def downTree (power : Nat) : Tree :=
  ((downOffsets power).foldl
    (fun values offset => fun lane => .node (values lane) (values (lane+offset)))
    (fun lane => .leaf lane)) 0

/-- Every CUDA block axis produced by the FP32 reduction configuration is a
power of two bounded by 512. Each permitted tree has complete leaf coverage.
This finite structural check does not evaluate floating-point arithmetic. -/
def downPlan (power : Nat) (bounded : power≤9) : Plan (2^power) where
  tree := downTree power
  positive := by positivity
  covers := by
    interval_cases power <;> decide +kernel

def downProgram {α : Type} (op : α → α → α) (input : Nat → α) (power : Nat) : α :=
  ((downOffsets power).foldl
    (fun values offset => fun lane => op (values lane) (values (lane+offset))) input) 0

private theorem down_run_stages {α : Type} (op : α → α → α) (seed : α) (input : Nat → α)
    (offsets : List Nat) (initial : Nat → Tree) (lane : Nat) :
    (offsets.foldl (fun values offset => fun t => Reduction.Tree.node (values t) (values (t+offset))) initial lane).run op seed input=
      (offsets.foldl (fun values offset => fun t => op (values t) (values (t+offset))))
        (fun t => (initial t).run op seed input) lane := by
  induction offsets generalizing initial with
  | nil => rfl
  | cons offset rest ih => exact ih (fun t => .node (initial t) (initial (t+offset)))

theorem down_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) (power : Nat) :
    (downTree power).run op seed input=downProgram op input power := by
  exact down_run_stages op seed input (downOffsets power) (fun i => .leaf i) 0

#axiomsClean down_exact
#axiomsClean downPlan
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
