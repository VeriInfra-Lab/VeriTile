import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LargeProfile

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Reduction
noncomputable section
set_option maxHeartbeats 10000000

/-- Runtime-aligned SM90 layout: four adjacent elements per thread, then
the next 512-element tile. Maximum and exponential-sum register reductions
use a seedless left fold. The smoothing sum has a fused prefix and is not
bound by this plain tree; `FusedReduction` supplies its separate lowering. -/
def alignedCoordinate (lane slot : Nat) : Nat := 4*lane+slot%4+512*(slot/4)
def alignedChain (lane : Nat) : Nat → Tree
  | 0 => .leaf (alignedCoordinate lane 0)
  | n+1 => .node (alignedChain lane n) (.leaf (alignedCoordinate lane (n+1)))

theorem aligned_chain_leaves (lane n : Nat) :
    (alignedChain lane n).leaves=(List.range (n+1)).map (alignedCoordinate lane) := by
  induction n with
  | zero => simp [alignedChain,Tree.leaves]
  | succ n ih => simp only [alignedChain,Tree.leaves,ih,List.range_succ,List.map_append,List.map_cons,List.map_nil]

theorem aligned_range :
    ((List.range 128).flatMap (fun lane => (List.range 512).map (alignedCoordinate lane))).Perm
      (List.range 65536) := by
  let pairs := (List.range 128).product (List.range 512)
  let index : Nat × Nat → Nat := fun pair => alignedCoordinate pair.1 pair.2
  have distinct : (pairs.map index).Nodup := by
    apply ((List.nodup_range (n:=128)).product (List.nodup_range (n:=512))).map_on
    intro a ha b hb equal
    have ah := List.mem_product.mp ha
    have bh := List.mem_product.mp hb
    simp only [List.mem_range] at ah bh
    apply Prod.ext
    all_goals dsimp only [index,alignedCoordinate] at equal
    all_goals omega
  have perm : (pairs.map index).Perm (List.range 65536) := by
    apply (List.perm_ext_iff_of_nodup distinct List.nodup_range).mpr
    intro i
    simp only [List.mem_map,List.mem_range]
    constructor
    · rintro ⟨pair,member,equal⟩
      have bounds := List.mem_product.mp member
      simp only [List.mem_range] at bounds
      dsimp only [index,alignedCoordinate] at equal
      omega
    · intro inside
      refine ⟨((i%512)/4,4*(i/512)+i%4),?_,?_⟩
      · apply List.mem_product.mpr
        simp only [List.mem_range]
        omega
      · dsimp only [index,alignedCoordinate]
        omega
  simpa [pairs,index,List.product,List.map_flatMap,List.map_map,Function.comp_def] using perm

def sm90AlignedLarge : Plan 65536 where
  tree := substitute (fun lane => alignedChain lane 511) lane128.tree
  positive := by decide
  covers := by
    rw [substitute_leaves]
    have leaves : (fun lane => (alignedChain lane 511).leaves)=
        (fun lane => (List.range 512).map (alignedCoordinate lane)) :=
      funext (fun lane => aligned_chain_leaves lane 511)
    rw [leaves]
    exact (lane128.covers.flatMap_right _).trans aligned_range

theorem sm90_aligned_large_exact {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) :
    sm90AlignedLarge.tree.run op seed input=
      sm90Four.tree.run op seed (fun warp => (warpTree warp 0).run op seed
        (fun lane => (alignedChain lane 511).run op seed input)) := by
  change (substitute (fun lane => alignedChain lane 511)
    (substitute (fun warp => warpTree warp 0) sm90Four.tree)).run op seed input=_
  rw [substitute_run,substitute_run]

#axiomsClean aligned_range
#axiomsClean sm90_aligned_large_exact
#axiomsClean sm90AlignedLarge
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
