import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealBackwardOuter
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def queryPrefix (p : BackwardParameters) (s : State R) (μ : RegionName → Nat → ℝ)
    (initial : Nat → BLane p → ℝ) (n b : Nat) (i : BLane p) : ℝ :=
  initial b i+∑ k ∈ Finset.range n,if k≤b then queryContribution p s μ k b i else 0

theorem rounds_reads (p : BackwardParameters) (s : State R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p s.mem μ) (sep : BackwardSeparate p) (n : Nat) :
    BackwardReads p (backwardRounds model p s n) μ := by
  induction n with
  | zero=>exact h
  | succ n ih=>exact outer_reads p s _ μ ih sep n

theorem rounds_dq (p : BackwardParameters) (s : State R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p s.mem μ) (sep : BackwardSeparate p)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (initial : Nat → BLane p → ℝ)
    (hq : ∀ b, b<p.blocks → ∀ i,s.mem p.dq (backwardOffset p s b i)=.mk .fp32 (some (initial b i)))
    (n : Nat) (hn : n≤p.blocks) (b : Nat) (hbn : b<p.blocks) (i : BLane p) :
    backwardRounds model p s n p.dq (backwardOffset p s b i)=
      .mk .fp32 (some (queryPrefix p s μ initial n b i)) := by
  induction n generalizing b i with
  | zero=>simpa [backwardRounds,queryPrefix] using hq b hbn i
  | succ n ih=>
    rw [backwardRounds,outer_dq p s _ μ (rounds_reads p s μ h sep n) sep n (by omega) hb
      (queryPrefix p s μ initial n) (by intro b hb i;exact ih (by omega) b hb i) b hbn i]
    simp [queryPrefix,Finset.sum_range_succ,add_assoc]

theorem rounds_kv (p : BackwardParameters) (s : State R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p s.mem μ) (sep : BackwardSeparate p)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (n : Nat) (hn : n≤p.blocks) (b : Nat) (hbn : b<n) (i : BLane p) :
    backwardRounds model p s n p.dk (backwardOffset p s b i)=
      .mk p.dtype.toTileDType (ofFloat p.dtype (some
        (∑ t ∈ Finset.range (p.blocks-b),keyContribution p s μ b (b+t) i))) ∧
    backwardRounds model p s n p.dv (backwardOffset p s b i)=
      .mk p.dtype.toTileDType (ofFloat p.dtype (some
        (∑ t ∈ Finset.range (p.blocks-b),valueContribution p s μ b (b+t) i))) := by
  induction n with
  | zero=>omega
  | succ n ih=>
    by_cases he : b=n
    · subst b
      exact ⟨outer_dk p s _ μ (rounds_reads p s μ h sep n) sep n (by omega) hb i,
        outer_dv p s _ μ (rounds_reads p s μ h sep n) sep n (by omega) hb i⟩
    · simp only [backwardRounds]
      rw [outer_otherKey p s _ sep n b he p.dk (Or.inl rfl) i,
        outer_otherKey p s _ sep n b he p.dv (Or.inr rfl) i]
      exact ih (by omega) (by omega)

/-- Actual shared-source backward execution and final typed stores. The
right-hand sides are explicit finite sums, not the execution recurrence. -/
theorem backward_execution_real (p : BackwardParameters) (s : State R)
    (μ : RegionName → Nat → ℝ) (h : BackwardReads p s.mem μ) (sep : BackwardSeparate p)
    (hBM : 0<p.blockM)
    (hb : ∀ b<p.blocks,∀ j : Fin p.blockM,BackwardRowBound p s b j)
    (hq : ∀ b, b<p.blocks → ∀ i,s.mem p.dq (backwardOffset p s b i)=.mk .fp32 (some 0)) :
    ∃ final,FP.Structural.exec (backend algebra model) (backwardSource p) s=some final ∧
      (∀ b, b<p.blocks → ∀ i : BLane p,
        final.mem p.dq (backwardOffset p s b i)=.mk .fp32 (some
          (∑ k ∈ Finset.range p.blocks,if k≤b then queryContribution p s μ k b i else 0)) ∧
        final.mem p.dk (backwardOffset p s b i)=.mk p.dtype.toTileDType (ofFloat p.dtype (some
          (∑ t ∈ Finset.range (p.blocks-b),keyContribution p s μ b (b+t) i))) ∧
        final.mem p.dv (backwardOffset p s b i)=.mk p.dtype.toTileDType (ofFloat p.dtype (some
          (∑ t ∈ Finset.range (p.blocks-b),valueContribution p s μ b (b+t) i)))) ∧
      (∀ r o,¬backwardWritten p s r o → final.mem r o=s.mem r o) := by
  obtain ⟨final,he,hm,_,hf⟩:=backwardCorrectness algebra model p s hBM
  refine ⟨final,he,?_,hf⟩
  intro b hbn i
  rw [hm]
  refine ⟨?_,rounds_kv p s μ h sep hb p.blocks (by omega) b hbn i⟩
  simpa [backwardFinalMemory,queryPrefix] using rounds_dq p s μ h sep hb (fun _ _=>0) hq p.blocks (by omega) b hbn i

#axiomsClean backward_execution_real
#specNonCircular queryPrefix avoiding [FP.Structural.exec,backwardRounds,backwardFinalMemory]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
