import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Physical
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Source

structure Owner where
  x : Nat
  y : Nat
  head : Nat
  deriving DecidableEq

/-- Actual program IDs recovered from a valid store's logical address. -/
def storeOwner (c : Shape) (a : Address) : Owner := match a.region with
  | .history | .finalState | .dhistory | .initialGradient => ⟨a.row/c.BK,a.col/c.BV,a.head⟩
  | .output => ⟨a.col/c.BV,a.row/64,a.head⟩
  | .dq | .dk => ⟨a.col/c.BK,a.row/64,a.head⟩
  | .dvPart => ⟨a.slice,a.row/64,a.head⟩
  | _ => ⟨0,0,a.head⟩

theorem tile_coordinates (tag : Region) (h s rb cb nr nc rows cols : Nat) (a : Address)
    (ha : a∈tile tag h s rb cb nr nc rows cols) :
    ∃ i<nr, ∃ j<nc, a=⟨tag,h,s,rb+i,cb+j⟩ := by
  simp only [tile,List.mem_flatMap,List.mem_range,List.mem_filterMap] at ha
  rcases ha with ⟨i,hi,j,hj,eq⟩
  split_ifs at eq with hab
  · exact ⟨i,hi,j,hj,(Option.some.inj eq).symm⟩

theorem tiled_div (block width lane : Nat) (hw : 0<width) (hl : lane<width) :
    (block*width+lane)/width=block := by
  rw [Nat.add_comm,Nat.add_mul_div_right lane block hw,Nat.div_eq_of_lt hl,Nat.zero_add]

theorem state_store_owner (c : Shape) (tag : Region)
    (htag : tag=.history ∨ tag=.finalState ∨ tag=.dhistory ∨ tag=.initialGradient)
    (h s kb vb : Nat) (a : Address)
    (ha : a∈tile tag h s (kb*c.BK) (vb*c.BV) c.BK c.BV c.K c.V) :
    storeOwner c a=⟨kb,vb,h⟩ := by
  rcases tile_coordinates _ _ _ _ _ _ _ _ _ a ha with ⟨i,hi,j,hj,rfl⟩
  rcases htag with rfl|rfl|rfl|rfl <;>
    simp [storeOwner,tiled_div kb c.BK i c.key_tile_pos hi,
      tiled_div vb c.BV j c.value_tile_pos hj]

theorem output_store_owner (c : Shape) (h chunk vb : Nat) (a : Address)
    (ha : a∈tile .output h 0 (chunk*64) (vb*c.BV) 64 c.BV c.T c.V) :
    storeOwner c a=⟨vb,chunk,h⟩ := by
  rcases tile_coordinates _ _ _ _ _ _ _ _ _ a ha with ⟨i,hi,j,hj,rfl⟩
  simp [storeOwner,tiled_div chunk 64 i (by decide) hi,tiled_div vb c.BV j c.value_tile_pos hj]

theorem key_store_owner (c : Shape) (tag : Region) (htag : tag=.dq ∨ tag=.dk)
    (h chunk kb : Nat) (a : Address)
    (ha : a∈tile tag h 0 (chunk*64) (kb*c.BK) 64 c.BK c.T c.K) :
    storeOwner c a=⟨kb,chunk,h⟩ := by
  rcases tile_coordinates _ _ _ _ _ _ _ _ _ a ha with ⟨i,hi,j,hj,rfl⟩
  rcases htag with rfl|rfl <;>
    simp [storeOwner,tiled_div chunk 64 i (by decide) hi,tiled_div kb c.BK j c.key_tile_pos hj]

theorem partial_store_owner (c : Shape) (h chunk kb vb : Nat) (a : Address)
    (ha : a∈tile .dvPart h kb (chunk*64) (vb*c.BV) 64 c.BV c.T c.V) :
    storeOwner c a=⟨kb,chunk,h⟩ := by
  rcases tile_coordinates _ _ _ _ _ _ _ _ _ a ha with ⟨i,hi,j,hj,rfl⟩
  simp [storeOwner,tiled_div chunk 64 i (by decide) hi]

/-- Equal physical active-write locations imply the same program owner.
The four preceding lemmas instantiate this property with all kernel grids. -/
theorem no_cross_program_write (c : Shape) (b : Binding c) (a a' : Address)
    (ha : a.Valid c) (ha' : a'.Valid c) (writeOnly : a.region∉inputRegions)
    (same : b.address a=b.address a') : storeOwner c a=storeOwner c a' := by
  exact congrArg (storeOwner c) (b.injective a a' ha ha' (Or.inl writeOnly) same)

#axiomsClean state_store_owner
#axiomsClean output_store_owner
#axiomsClean key_store_owner
#axiomsClean partial_store_owner
#axiomsClean no_cross_program_write
end Source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

