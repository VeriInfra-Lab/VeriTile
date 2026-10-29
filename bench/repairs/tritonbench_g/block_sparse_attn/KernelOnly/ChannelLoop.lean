import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Channel

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem scoreAccumulation_succ {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial : State (Replicas α)) (column : Int) (block : Nat) :
    scoreAccumulation a native p initial column (block+1) =
      scoreStep a native p initial column
        (scoreAccumulation a native p initial column block) block := by
  simp [scoreAccumulation,List.range_succ,List.foldl_append]

/-- Every channel block is visited, with the exact live score tile retained
between fixed K8 instruction chains. -/
theorem channelLoopExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (column : Int) (block : Nat) (bound : block≤p.blocks)
    (h : ChannelInvariant a p initial carry column
      (scoreAccumulation a native p initial column block) current) :
    ∃ final, loop a.model "score_block" block p.blocks (channelBody native p) current=some final ∧
      ChannelInvariant a p initial carry column
        (scoreAccumulation a native p initial column p.blocks) final := by
  have aux : ∀ n k (current : State (Replicas α)), p.blocks-k=n → k≤p.blocks →
      ChannelInvariant a p initial carry column (scoreAccumulation a native p initial column k) current →
      ∃ final, loop a.model "score_block" k p.blocks (channelBody native p) current=some final ∧
        ChannelInvariant a p initial carry column (scoreAccumulation a native p initial column p.blocks) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases less : k<p.blocks
      · let next := channelExecution a native p initial current carry column
          (scoreAccumulation a native p initial column k) k h
        have hi : ChannelInvariant a p initial carry column
            (scoreAccumulation a native p initial column (k+1)) next.val := by
          rw [scoreAccumulation_succ]
          exact channelInvariant a native p initial current carry column
            (scoreAccumulation a native p initial column k) k h
        obtain ⟨final,he,hf⟩ := ih (p.blocks-(k+1)) (by omega) (k+1) next.val rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [loop,if_pos less,next.property]
        exact he
      · have equal : k=p.blocks := by omega
        subst k
        exact ⟨current,by rw [loop,if_neg (by omega)],h⟩
  exact aux _ _ _ rfl bound h

#axiomsClean scoreAccumulation_succ
#axiomsClean channelLoopExecution
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
