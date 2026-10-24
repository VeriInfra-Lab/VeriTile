import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Storage

namespace VeriTile.Bench.Repairs.KldivOps.Storage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
noncomputable section

/-- All four bytes are retained, including every NaN payload. -/
def Lossless {Word : Type} (codec : Codec Word) : Prop := ∀ bytes,codec.encode (codec.decode bytes)=bytes

theorem fp32_lossless : Lossless VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec.littleEndian := by
  intro bytes
  funext k
  apply Fin.ext
  have b0 := (bytes 0).isLt
  have b1 := (bytes 1).isLt
  have b2 := (bytes 2).isLt
  have b3 := (bytes 3).isLt
  fin_cases k <;>
    simp only [VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec.littleEndian,
      VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec.encode,
      VeriTile.Bench.Repairs.DiagSsmTriton.WordCodec.decode,FP.Value.ofBits,FP.Value.raw,BitVec.toNat_ofNat]
  all_goals norm_num [FP.Format.width,FP.Format.fractionBits] at *
  all_goals first | change _ = (bytes 0).val | change _ = (bytes 1).val | change _ = (bytes 2).val | change _ = (bytes 3).val
  all_goals omega

theorem word_bytes {Word : Type} (codec : Codec Word) (lossless : Lossless codec)
    (a b : Heap) (address : Nat) (same : load codec a address=load codec b address) (k : Fin 4) :
    a (address+k.val)=b (address+k.val) := by
  have h := congrArg (fun w=>codec.encode w k) same
  unfold Lossless at lossless
  simpa only [load,lossless] using h

/-- Committing the final typed state is exactly the sequence of physical
stores in the active prefix. Unwritten work tail bytes remain unchanged; this
is proved from the actual final source cells and concrete lossless storage. -/
theorem commit_prefix {Word : Type} [Inhabited Word] (codec : Codec Word) (lossless : Lossless codec)
    (a : Arena) (state : State Word) (m : Heap) (count : Nat) (capacity : count≤a.output.words)
    (tail : ∀ i,count ≤ i → i<a.output.words → (state.mem "O" i).read .real=load codec m (a.output.address i)) :
    commit codec a state m=writeValues codec a.output (List.range count)
      (fun i => (state.mem "O" i).read .real) m := by
  funext p
  by_cases inside : ∃ i,i<a.output.words ∧ a.output.address i≤p ∧ p<a.output.address i+4
  · obtain ⟨i,hi,lo,up⟩ := inside
    let k : Fin 4 := ⟨p-a.output.address i,by omega⟩
    have hp : a.output.address i+k.val=p := by dsimp [k]; omega
    have same : load codec (commit codec a state m) (a.output.address i)=
        load codec (writeValues codec a.output (List.range count) (fun j => (state.mem "O" j).read .real) m)
          (a.output.address i) := by
      rw [commit_read codec a state m i hi]
      by_cases active : i<count
      · rw [writeValues_read codec a.output (List.range count) _ m i (List.mem_range.mpr active)]
      · rw [writeValues_load_outside codec a.output (List.range count) _ m i (by
          simpa only [List.mem_range] using active)]
        exact tail i (by omega) hi
    simpa only [hp] using word_bytes codec lossless _ _ (a.output.address i) same k
  · have outside : ∀ i,i<a.output.words → p<a.output.address i ∨ a.output.address i+4≤p := by
      intro i hi
      by_contra h
      exact inside ⟨i,hi,by omega,by omega⟩
    rw [commit_padding codec a state m p outside]
    symm
    apply writeValues_frame
    intro i hi
    exact outside i (by have := List.mem_range.mp hi; omega)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Storage
