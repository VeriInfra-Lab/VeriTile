import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ProfileWitness

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
noncomputable section
set_option maxHeartbeats 8000000

def substitute (inner : Nat → Tree) : Tree → Tree
  | .leaf i => inner i
  | .seed => .seed
  | .node a b => .node (substitute inner a) (substitute inner b)

theorem substitute_leaves (inner : Nat → Tree) (tree : Tree) :
    (substitute inner tree).leaves=tree.leaves.flatMap (fun i => (inner i).leaves) := by
  induction tree with
  | leaf i => simp [substitute,Tree.leaves]
  | seed => rfl
  | node a b ia ib => simp [substitute,Tree.leaves,ia,ib]

theorem substitute_run {α : Type} (op : α → α → α) (seed : α) (input : Nat → α)
    (inner : Nat → Tree) (tree : Tree) :
    (substitute inner tree).run op seed input=tree.run op seed (fun i => (inner i).run op seed input) := by
  induction tree with
  | leaf i => rfl
  | seed => rfl
  | node a b ia ib => simp only [substitute,Tree.run,ia,ib]

/-- Register reduction: the first two data registers are combined, then
each subsequent data register is combined into the previous accumulator.
No identity seed is silently inserted. -/
def stridedChain (lane stride : Nat) : Nat → Tree
  | 0 => .leaf lane
  | n+1 => .node (stridedChain lane stride n) (.leaf (lane+stride*(n+1)))

theorem strided_chain_leaves (lane stride n : Nat) :
    (stridedChain lane stride n).leaves=(List.range (n+1)).map (fun i => lane+stride*i) := by
  induction n with
  | zero => simp [stridedChain,Tree.leaves]
  | succ n ih => simp only [stridedChain,Tree.leaves,ih,List.range_succ,List.map_append,List.map_cons,List.map_nil]

def warpTree (warp lane : Nat) : Tree :=
  ([16,8,4,2,1] : List Nat).foldl
    (fun values offset => fun t => .node (values t) (values (Nat.xor t offset)))
    (fun t => .leaf (32*warp+t)) lane

/-- Four warp leaders write their scalar into shared memory. Threads 0..3
load those four words, use butterfly offsets 2 then 1, and thread 0 stores
the scalar subsequently broadcast by the shared load. -/
def lane128 : Plan 128 :=
  ⟨substitute (fun warp => warpTree warp 0) sm90Four.tree,by decide +kernel,by decide⟩

theorem transposed_range (count : Nat) :
    ((List.range 128).flatMap (fun lane => (List.range count).map (fun i => lane+128*i))).Perm
      (List.range (128*count)) := by
  let pairs := (List.range 128).product (List.range count)
  let index : Nat × Nat → Nat := fun pair => pair.1+128*pair.2
  have distinct : (pairs.map index).Nodup := by
    apply ((List.nodup_range (n:=128)).product (List.nodup_range (n:=count))).map_on
    intro a ha b hb equal
    have ah := List.mem_product.mp ha
    have bh := List.mem_product.mp hb
    simp only [List.mem_range] at ah bh
    apply Prod.ext
    all_goals dsimp only [index] at equal
    all_goals omega
  have perm : (pairs.map index).Perm (List.range (128*count)) := by
    apply (List.perm_ext_iff_of_nodup distinct List.nodup_range).mpr
    intro i
    simp only [List.mem_map,List.mem_range]
    constructor
    · rintro ⟨pair,member,equal⟩
      have bounds := List.mem_product.mp member
      simp only [List.mem_range] at bounds
      dsimp only [index] at equal
      omega
    · intro inside
      refine ⟨(i%128,i/128),?_,?_⟩
      · apply List.mem_product.mpr
        simp only [List.mem_range]
        omega
      · dsimp only [index]
        omega
  simpa [pairs,index,List.product,List.map_flatMap,List.map_map,Function.comp_def] using perm

/-- Complete native 65536-leaf tree: 512 registers per thread, 128 threads,
then warp shuffles and the four-word shared-memory reduction. -/
def sm90Large : Plan 65536 where
  tree := substitute (fun lane => stridedChain lane 128 511) lane128.tree
  positive := by decide
  covers := by
    rw [substitute_leaves]
    have leaves : (fun lane => (stridedChain lane 128 511).leaves)=
        (fun lane => (List.range 512).map (fun i => lane+128*i)) :=
      funext (fun lane => strided_chain_leaves lane 128 511)
    rw [leaves]
    exact (lane128.covers.flatMap_right _).trans (transposed_range 512)

theorem sm90_large_covers : sm90Large.tree.leaves.Perm (List.range 65536) := sm90Large.covers

/-- The staged register/warp/shared schedule evaluates exactly this complete
tree for any scalar reducer. No associativity, seed-identity, or entire
reduction-result assumption is used. -/
theorem sm90_large_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) :
    sm90Large.tree.run op seed input=
      sm90Four.tree.run op seed (fun warp => (warpTree warp 0).run op seed
        (fun lane => (stridedChain lane 128 511).run op seed input)) := by
  change (substitute (fun lane => stridedChain lane 128 511)
    (substitute (fun warp => warpTree warp 0) sm90Four.tree)).run op seed input=_
  rw [substitute_run,substitute_run]

#axiomsClean substitute_leaves
#axiomsClean substitute_run
#axiomsClean strided_chain_leaves
#axiomsClean transposed_range
#axiomsClean sm90_large_covers
#axiomsClean sm90_large_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
