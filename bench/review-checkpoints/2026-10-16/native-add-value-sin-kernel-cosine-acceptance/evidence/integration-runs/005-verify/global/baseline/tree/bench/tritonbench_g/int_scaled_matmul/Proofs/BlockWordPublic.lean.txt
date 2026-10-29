import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordWitness
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section

/-- A source load has a genuine signed-byte payload. In-bounds payloads are
exact initial stored bytes; outside payloads retain the unspecified oracle. -/
def InputReadback {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape) : Prop :=
  ∃ w : BitVec 8, loaded s ptr i = w.toInt ∧
    (ptr.inBounds (TileShape.indexToList shape i) [0,1] = Bool.true →
      s.mem ptr.region (ptr.address (TileShape.indexToList shape i)) = .mk .int w.toInt)

theorem storage_readback {α : Type} [Inhabited α] {shape : TileShape}
    (s : State α) (ptr : BlockPtr) (i : TileIndex shape) (h : ByteStorage s ptr i) : InputReadback s ptr i := by
  by_cases hi : ptr.inBounds (TileShape.indexToList shape i) [0,1] = Bool.true
  · obtain ⟨w, hm, hv⟩ := in_bounds_byte s ptr i h hi
    exact ⟨w, hv, fun _ => hm⟩
  · exact ⟨loadedBits s ptr i, loaded_bits s ptr i, fun hc => False.elim (hi hc)⟩

/-- Candidate for the public specification: full raw K loop including every
unspecified tail lane, independent initial-state output and a complete frame. -/
theorem machine_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (hBK : 0 < p.blockInner) (hBytes : ByteInputs p s)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (backend M) (source p) s = some final ∧
      Post p s final (outputValue p s) ∧
      (∀ k, k < steps p → ∀ i : AIndex p, InputReadback s (aPointer p s k) i) ∧
      (∀ k, k < steps p → ∀ i : BIndex p, InputReadback s (bPointer p s k) i) := by
  obtain ⟨final, he, hp⟩ := correctness M p hBK s hw
  exact ⟨final, he, hp,
    fun k hk i => storage_readback s _ i (hBytes.1 k hk i),
    fun k hk i => storage_readback s _ i (hBytes.2 k hk i)⟩

#axiomsClean storage_readback
#axiomsClean machine_correctness
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
