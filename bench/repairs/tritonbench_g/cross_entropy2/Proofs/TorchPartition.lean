import bench.repairs.tritonbench_g.cross_entropy2.Proofs.TorchThread
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LargeProfile

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
noncomputable section
set_option maxHeartbeats 10000000

def threadIndices (count threads slot : Nat) : List Nat :=
  (List.range count).filter (fun i => i%threads==slot)

/-- Strided input assignment partitions a reduction even when some worker
threads have no data. Such workers retain their seed-only accumulators. -/
theorem thread_indices_cover (count threads : Nat) (positive : 0<threads) :
    ((List.range threads).flatMap (threadIndices count threads)).Perm (List.range count) := by
  have distinct : ((List.range threads).flatMap (threadIndices count threads)).Nodup := by
    apply List.nodup_flatMap.mpr
    refine ⟨fun slot _ => List.nodup_range.filter _,?_⟩
    apply List.nodup_range.imp
    intro a b different i ha hb
    have ia := (List.mem_filter.mp ha).2
    have ib := (List.mem_filter.mp hb).2
    simp only [beq_iff_eq] at ia ib
    exact different (ia.symm.trans ib)
  apply (List.perm_ext_iff_of_nodup distinct List.nodup_range).mpr
  intro i
  simp only [List.mem_flatMap,threadIndices,List.mem_filter,List.mem_range,beq_iff_eq]
  constructor
  · rintro ⟨_,_,inside,_⟩
    exact inside
  · intro inside
    exact ⟨i%threads,Nat.mod_lt i positive,inside,rfl⟩

def threadTree (indices : List Nat) : Tree :=
  substitute (fun i => .leaf (indices[i]?.getD 0)) (threadFourTree indices.length)

theorem thread_tree_covers (indices : List Nat) : (threadTree indices).leaves.Perm indices := by
  have all : (threadFourTree indices.length).leaves.Perm (List.range indices.length) := by
    simpa only [threadFourTree,Tree.leaves,append_leaves,List.nil_append] using accumulator_indices_cover indices.length
  have mapped := all.map (fun i => indices[i]?.getD 0)
  have reads : (List.range indices.length).map (fun i => indices[i]?.getD 0)=indices := by
    apply List.ext_getElem
    · simp
    · intro i hi hj
      simp [List.getElem?_eq_getElem hj]
  rw [reads] at mapped
  simpa only [threadTree,substitute_leaves,Tree.leaves,← List.map_eq_flatMap] using mapped

/-- Complete four-accumulator workers followed by an explicit inter-worker
reduction tree. This construction models block and CTA stages without any
associativity, identity, or reduction-result premise. -/
def partitionPlan (count : Nat) (positive : 0<count) {threads : Nat} (outer : Plan threads) : Plan count where
  tree := substitute (fun slot => threadTree (threadIndices count threads slot)) outer.tree
  positive := positive
  covers := by
    rw [substitute_leaves]
    exact (outer.covers.flatMap (fun slot _ => thread_tree_covers (threadIndices count threads slot))).trans
      (thread_indices_cover count threads outer.positive)

theorem partition_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α)
    (count : Nat) (positive : 0<count) {threads : Nat} (outer : Plan threads) :
    (partitionPlan count positive outer).tree.run op seed input=
      outer.tree.run op seed (fun slot =>
        threadFourProgram op seed (fun i => input ((threadIndices count threads slot)[i]?.getD 0))
          (threadIndices count threads slot).length) := by
  simp only [partitionPlan,threadTree,substitute_run,thread_four_exact,Tree.run]

#axiomsClean thread_indices_cover
#axiomsClean thread_tree_covers
#axiomsClean partition_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
