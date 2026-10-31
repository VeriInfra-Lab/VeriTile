import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_RealSafety
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_IO

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Flat
open VeriTile.Triton Math Real
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ ∃ i : TileIndex [p.BT,p.BV],
    active p l i.1.val i.2.1.val ∧ cell.2 = alloc.addr r.output (outputAddress p l i.1.val i.2.1.val)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (s t : BlockState)
    (hf : BlockState.WriteWithin (Real.footprint p l r) s t) :
    BlockState.WriteWithin (footprint alloc p l r) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨rg,k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,i,hi,haddr⟩
      apply hn
      refine ⟨rfl,i,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Flat
