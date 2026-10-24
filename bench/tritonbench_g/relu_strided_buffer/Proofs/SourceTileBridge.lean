import bench.tritonbench_g.relu_strided_buffer.Proofs.SourceSignedBranch

/-!
The signed ordinary-pointer lowering agrees, by observable memory and padding,
with the complete explicit block-pointer per-tile body on the common natural
stride domain. The signed execution theorem additionally gives its own active
readback/frame contract for either stride sign. No source output oracle occurs.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceTileBridge
open VeriTile.Triton SignedView SignedTile SourceBranch

def signedOfNat (c : NatViews) : SourceSignedBranch.SignedViews :=
  ⟨c.inputView, c.outputView, c.n, c.B⟩

/-- All index, pointer, value and store statements execute on both sides.
Registers intentionally differ: the signed primitive stores address vectors,
while the existing common BlockPtr stores natural-stride pointer metadata. -/
theorem natural_source_lowering (c : NatViews) (T : Nat) (s : BlockState)
    (ht : s.regs .nat [] "tile_id" = some (Tile.scalar T)) :
    ∃ blockFinal signedFinal,
      stepStmts c.perTile s = some blockFinal ∧
      stepStmts (signedOfNat c).perTile s = some signedFinal ∧
      ObsEq blockFinal signedFinal := by
  refine ⟨c.sourceFinal T s, (signedOfNat c).sourceFinal T s,
    sourceTile_exec c T s ht, SourceSignedBranch.sourceTile_exec (signedOfNat c) T s ht, ?_⟩
  exact (sourceFinal_obs c T s).trans
    (SourceSignedBranch.sourceFinal_obs (signedOfNat c) T s).symm

/-- Signed dynamic source tile: all active outputs are derived from this call's
initial input, and every other typed cell is preserved. Output validity and
nonzero stride are ordinary view requirements, with no sign or size cap. -/
theorem signed_source_tile_success (c : SourceSignedBranch.SignedViews)
    {capacity T : Nat} (hout : c.outputView.Valid c.n capacity)
    (hstride : c.outputView.stride ≠ 0) (s : BlockState)
    (ht : s.regs .nat [] "tile_id" = some (Tile.scalar T)) :
    ∃ final, stepStmts c.perTile s = some final ∧
      (∀ i : TileIndex [c.B], T * c.B + i.1.val < c.n →
        final.mem c.outputView.region (c.outputView.address (T * c.B + i.1.val)) =
          storedCell32 (relu32 (s.readMemValue .fp32 c.inputView.region
            (c.inputView.address (T * c.B + i.1.val))))) ∧
      (∀ r a, (r ≠ c.outputView.region ∨ ∀ i : TileIndex [c.B],
        T * c.B + i.1.val < c.n → a ≠ c.outputView.address (T * c.B + i.1.val)) →
        final.mem r a = s.mem r a) := by
  obtain ⟨static, hexec, hread, hframe⟩ :=
    tile_success c.inputView c.outputView (start := T * c.B) (B := c.B) hout hstride s
  rw [tile_exec] at hexec
  cases Option.some.inj hexec
  refine ⟨c.sourceFinal T s, SourceSignedBranch.sourceTile_exec c T s ht, ?_, ?_⟩
  · simpa only [(SourceSignedBranch.sourceFinal_obs c T s).1] using hread
  · simpa only [(SourceSignedBranch.sourceFinal_obs c T s).1] using hframe

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceTileBridge
