import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSource

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.DenseSafety
open VeriTile.Triton Geometry DenseSource
open VeriTile.Triton.Memory.AccessHelpers
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def bounds (c : Shape) : RegionBounds := fun r =>
  if r = "q" then c.querySize else if r = "k" then c.keySize else
  if r = "m" then c.normalizerSize else if r = "out" then c.outputSize else 0

theorem full_row (N B block lane : Nat) (hB : 0 < B)
    (hb : block < RealLoop.tripCount N B) (hl : lane < B) (he : N % B = 0) :
    block*B+lane < N := by
  have hd := digit_bound block (RealLoop.tripCount N B) lane B hb hl
  rwa [LogicalFormula.exact_cover N B hB he] at hd

theorem query (c : Shape) (z : Fin c.Z) (h : Fin c.H) (block : Nat)
    (hb : block < RealLoop.tripCount c.N c.B) :
    BlockBounds (bounds c)
      (RealLoads.queryPointer "q" c.B c.D c.N (c.queryBase z.val h.val) c.D 1 (block*c.B))
      (RealLoads.checks (decide (c.N % c.B = 0))) := by
  rintro ⟨row,d,⟨⟩⟩ ha
  have hr : block*c.B+row.val < c.N := by
    by_cases he : c.N % c.B = 0
    · exact full_row _ _ _ _ (supported_positive c.B c.block) hb row.isLt he
    · simpa [RealLoads.queryPointer, RealLoads.checks, he, BlockPtr.inBounds_2d_offsets,
        TileShape.indexToList, d.isLt] using ha
  have hn := query_bound c z h ⟨block*c.B+row.val,hr⟩ d
  simpa [RealLoads.queryPointer, BlockPtr.address_2d_offsets, TileShape.indexToList,
    bounds, Nat.add_mul, Nat.add_assoc] using hn

theorem key (c : Shape) (z : Fin c.Z) (h : Fin c.H) (block : Fin c.programs) :
    BlockBounds (bounds c)
      (RealLoads.keyPointer "k" c.B c.D c.NK (c.keyBase z.val h.val) c.D 1 block.val)
      (RealLoads.checks (decide (c.NK % c.B = 0))) := by
  rintro ⟨d,col,⟨⟩⟩ ha
  have hc : block.val*c.B+col.val < c.NK := by
    by_cases he : c.NK % c.B = 0
    · exact full_row _ _ _ _ (supported_positive c.B c.block) block.isLt col.isLt he
    · simpa [RealLoads.keyPointer, RealLoads.checks, he, BlockPtr.inBounds_2d_offsets,
        TileShape.indexToList, d.isLt] using ha
  have hn := key_bound c z h ⟨block.val*c.B+col.val,hc⟩ d
  simpa [RealLoads.keyPointer, BlockPtr.address_2d_offsets, TileShape.indexToList,
    bounds, Nat.add_mul, Nat.add_assoc] using hn

theorem all_bounds (c : Shape) (z : Fin c.Z) (h : Fin c.H) (block : Fin c.programs)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    RealSafety.Bounds (bounds c) (DenseSource.config c block.val z.val h.val offset size scale window complement)
      c.N "out" (c.outputBase z.val h.val) := by
  constructor
  · exact query c z h
  · exact key c z h block
  · intro row hr
    exact normalizer_bound c z h ⟨row,hr⟩
  · intro lane hl
    have hb := output_bound c z h ⟨block.val*c.B+lane.val,hl⟩
    simpa [Nat.add_assoc] using hb

/-- Every active load and store of every actual dense-grid program lies in
the finite host allocation. Inactive M lanes need no allocated cells. -/
theorem source_safe (c : Shape) (s : BlockState) (z : Fin c.Z) (h : Fin c.H)
    (block : Fin c.programs) (offset size : Int) (scale : ℝ) (window complement : Bool) (hOffsets : KeyOffsetsFit c) :
    (DenseSource.kernel c offset size scale window complement).eraseDType.toAlgKernel.TraceSafe
      (bounds c) (programState c s block.val z.val h.val) := by
  apply RealSource.safe (bounds c) (programState c s block.val z.val h.val) c.B c.D
    (parameters c offset size scale window complement) Bool.false (supported_positive c.B c.block)
    (by simpa [programState] using hOffsets block)
  rw [config_eq]
  change RealSafety.Bounds _ _ c.N "out"
    (RealSource.outputBase (parameters c offset size scale window complement) (z.val*c.H+h.val))
  rw [output_base]
  exact all_bounds c z h block offset size scale window complement

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.DenseSafety
