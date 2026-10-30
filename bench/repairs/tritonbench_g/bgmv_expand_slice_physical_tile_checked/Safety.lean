import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Memory
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedProgram
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton Contract Direct
noncomputable section

/-- This is exactly the source's largest broadcast tile, required only when a
program exists. It is a frontend shape bound, not a device resource guarantee. -/
theorem static_tile (r : Request) (heap : ByteMemory) (h : Admitted r heap) (p : Program r heap h) :
    256*(call r heap h).BK≤2^20 ∧ (call r heap h).BK≤4096 := by
  have c:=checked r heap h
  have hb : p.batch<r.i.numel:=p.batchInside
  have hn : p.row<width r.w:=p.rowInside
  have ht:=c.tile
  rcases ht with empty|empty|bound
  · omega
  · omega
  · change 256*block r.w≤2^20 ∧ block r.w≤4096
    omega

theorem source_shapes (r : Request) (heap : ByteMemory) (h : Admitted r heap) (p : Program r heap h) :
    ∀ shape∈([[],[(call r heap h).BK],[256],[1,(call r heap h).BK],[256,1],
      [256,(call r heap h).BK]] : List (List Nat)),
      shape.prod≤2^20 := by
  have bound:=static_tile r heap h p
  intro shape member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl <;> simp_all <;> omega

theorem signed_cdiv (r : Request) (heap : ByteMemory) (h : Admitted r heap) (p : Program r heap h) :
    let n:=(call r heap h).N
    ((n:Int)+n-1)/n=1 ∧ 0≤(n:Int)+n-1 ∧ (n:Int)+n-1<2^63 := by
  have hp:=p.rowInside
  have bound:=(checked r heap h).dimensions.2.2.1
  refine ⟨Signed.Program.signed_length (call r heap h) (by omega),?_,?_⟩
  · dsimp only
    omega
  · change (Contract.width r.w:Int)+Contract.width r.w-1<2^63
    omega

theorem all_formed (r : Request) (heap : ByteMemory) (h : Admitted r heap) (p : Program r heap h)
    (active : indices r heap p.batch≠-1) (n : Fin 256) (k : Fin (Contract.block r.w)) :
    let d:=call r heap h
    let slot:=(indices r heap p.batch).toNat
    4*Bounds.inputOffset d p k.val<2^63 ∧
    2*Bounds.weightOffset d p slot n.val k.val<2^63 ∧
    4*Bounds.outputOffset d p n.val<2^63 := by
  have selected:=((Contract.selected_valid r.x r.w r.y r.i r.scalars (indices r heap) h
    p.batch p.row p.batchInside p.rowInside).resolve_left active).2
  exact Bounds.formed r.x r.w r.y r.i r.scalars (indices r heap) h p _ selected n k

#axiomsClean static_tile
#axiomsClean source_shapes
#axiomsClean signed_cdiv
#axiomsClean all_formed
#axiomsClean Bounds.signed_offsets
#specNonCircular Contract.StaticTile avoiding [Surface.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
