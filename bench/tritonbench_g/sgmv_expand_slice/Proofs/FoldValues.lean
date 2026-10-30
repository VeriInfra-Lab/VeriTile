import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceLoop

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)

theorem foldRange_range' {α : Type} (update : Nat → α → α) (start count : Nat) (initial : α) :
    foldRange update start (start+count) 1 initial =
      (List.range' start count).foldl (fun acc k => update k acc) initial := by
  induction count generalizing start initial with
  | zero => rw [foldRange]; simp
  | succ count ih =>
      rw [foldRange]
      simp only [Nat.one_ne_zero,show start < start+(count+1) by omega,↓reduceDIte,
        List.range'_succ,List.foldl_cons]
      simpa only [Nat.add_assoc,Nat.add_comm 1 count] using ih (start+1) (update start initial)

theorem foldRange_range {α : Type} (update : Nat → α → α) (count : Nat) (initial : α) :
    foldRange update 0 count 1 initial =
      (List.range count).foldl (fun acc k => update k acc) initial := by
  simpa only [Nat.zero_add,List.range_eq_range'] using foldRange_range' update 0 count initial

theorem foldRange_rel {α β : Type} (left : Nat → α → α) (right : Nat → β → β)
    (relation : α → β → Prop) (count : Nat)
    (step : ∀ k, k < count → ∀ a b, relation a b → relation (left k a) (right k b))
    (a : α) (b : β) (initial : relation a b) :
    relation (foldRange left 0 count 1 a) (foldRange right 0 count 1 b) := by
  rw [foldRange_range,foldRange_range]
  apply List.foldl_rel initial
  intro k hk x y hxy
  exact step k (List.mem_range.mp hk) x y hxy

#axiomsClean foldRange_range'
#axiomsClean foldRange_range
#axiomsClean foldRange_rel
end VeriTile.Bench.Repairs.SgmvExpandSlice
