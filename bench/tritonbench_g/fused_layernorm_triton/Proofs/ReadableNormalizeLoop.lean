import bench.tritonbench_g.fused_layernorm_triton.Proofs.ReadableNormalize

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableNormalizeLoop
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open ReadableNormalize
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem chunk_effect (hoist : Bool) (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s)
    (pins : hoist = Bool.true → Pins s row mean inverse) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .int [] (fun _ => (k*B : Nat))) = some final ∧
      NormalizeEffect M s final X W Bias Y D B k 1 row mean inverse ∧ SourceIndexPins B final ∧
      (hoist = Bool.true → Pins final row mean inverse) := by
  have chunk : ∃ final, run (withEquality M equal32)
      (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .int [] (fun _ => (k*B : Nat))) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse ∧ SourceIndexPins B final ∧
      (hoist = Bool.true → Pins final row mean inverse) := by
    cases hoist
    · obtain ⟨t,ht,hmemory,hregs,hindex⟩ := original_chunk_run M equal32 s X W Bias Y Mean Rstd S D B k row tree mean inverse hm hr hx indexPins
      exact ⟨t,ht,hmemory,hregs,hindex,by simp⟩
    · obtain ⟨t,ht,hmemory,hregs,hindex,hpin⟩ := candidate_chunk_run M equal32 s X W Bias Y Mean Rstd S D B k row tree mean inverse hm hr hx indexPins (pins rfl)
      exact ⟨t,ht,hmemory,hregs,hindex,fun _ => hpin⟩
  obtain ⟨t,ht,hmem,hreg,hindex,hpin⟩ := chunk
  refine ⟨t,ht,⟨hreg,?_,?_⟩,hindex,hpin⟩
  · intro l hl hu i hi
    have heq : l=k := by omega
    subst l
    rw [hmem]
    have hv := State.masked_scatter_readback Y
      (fun i : TileIndex [B] => k*B+i.1.val+D*row)
      (fun i => Cell.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k*B+i.1.val) mean inverse))
      (fun i => k*B+i.1.val<D) s (i,()) hi
      (by intro a b hab; dsimp at hab; exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _))
    simpa [outputChunk,maskedNormalizeValue,normalizeValue,hi] using hv
  · intro q off hmiss
    rw [hmem]
    apply State.masked_scatter_frame
    rcases hmiss with hq | ho
    · exact Or.inl hq
    · exact Or.inr (fun i _ hi => ho k (by omega) (by omega) i.1 hi)

theorem loop_run (hoist : Bool) (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B start count row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α) (hB : 0<B)
    (hc : count = blockCount D B-start)
    (hX : X≠Y) (hW : W≠Y) (hBias : Bias≠Y)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s)
    (pins : hoist = Bool.true → Pins s row mean inverse) :
    ∃ final, rangeInt (withEquality M equal32) "roffset" (start*B : Nat) D B
      (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body s = some final ∧
      NormalizeEffect M s final X W Bias Y D B start count row mean inverse ∧
      SourceIndexPins B final ∧ (hoist = Bool.true → Pins final row mean inverse) := by
  induction count generalizing start s with
  | zero =>
    have hdone : D≤start*B := by
      by_contra hd
      have hk := (ReadableStatistics.chunk_active D B start hB).mpr (by omega)
      omega
    refine ⟨s,?_,⟨⟨rfl,rfl,hm,hr,hx⟩,?_,?_⟩,indexPins,pins⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_neg (by exact_mod_cast not_lt.mpr hdone)]
    · intro k hl hu; omega
    · intro q off h; rfl
  | succ count ih =>
    have hk : start<blockCount D B := by omega
    have hactive := (ReadableStatistics.chunk_active D B start hB).mp hk
    obtain ⟨u,hu,he,huIndex,hpin⟩ := chunk_effect hoist M equal32 s X W Bias Y Mean Rstd S D B start row tree mean inverse hm hr hx indexPins pins
    obtain ⟨t,ht,hf,htIndex,hpins⟩ := ih u (start+1) (by omega) he.registers.mean he.registers.rstd he.registers.row huIndex hpin
    have hXu : u.mem X=s.mem X := funext (fun off => he.frame X off (.inl hX))
    have hWu : u.mem W=s.mem W := funext (fun off => he.frame W off (.inl hW))
    have hBu : u.mem Bias=s.mem Bias := funext (fun off => he.frame Bias off (.inl hBias))
    refine ⟨t,?_,⟨⟨hf.registers.pids.trans he.registers.pids,
      hf.registers.undef.trans he.registers.undef,hf.registers.mean,hf.registers.rstd,hf.registers.row⟩,?_,?_⟩,htIndex,hpins⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_pos (by exact_mod_cast hactive),hu]
      simpa only [Nat.add_mul,Nat.one_mul,Option.bind,Nat.cast_add] using ht
    · intro k hl hupp i hi
      by_cases hke : k=start
      · subst k
        rw [hf.frame Y (start*B+i.val+D*row) ?_]
        · exact he.values start (by omega) (by omega) i hi
        · right
          intro l hls hlu j hj heq
          have hs := (chunk_unique B start l i.val j.val i.isLt j.isLt (by omega)).1
          omega
      · have hv := hf.values k (by omega) (by omega) i hi
        simpa only [normalizeValue,hXu,hWu,hBu] using hv
    · intro q off hmiss
      rw [hf.frame q off ?_,he.frame q off ?_]
      · rcases hmiss with hq | ho
        · exact Or.inl hq
        · exact Or.inr (fun k hl hu i hi => ho k (by omega) (by omega) i hi)
      · rcases hmiss with hq | ho
        · exact Or.inl hq
        · exact Or.inr (fun k hl hu i hi => ho k (by omega) (by omega) i hi)

theorem whole_row (hoist : Bool) (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α) (hB : 0<B)
    (hX : X≠Y) (hW : W≠Y) (hBias : Bias≠Y)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s)
    (pins : hoist = Bool.true → Pins s row mean inverse) :
    ∃ final, rangeInt (withEquality M equal32) "roffset" 0 D B
      (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body s = some final ∧
      NormalizeRegisters s final row mean inverse ∧
      (∀ col,col<D → final.mem Y (col+D*row)=.mk .bf16 (normalizeValue M s X W Bias D row col mean inverse)) ∧
      (∀ q off,(q≠Y ∨ ∀ col,col<D → off≠col+D*row) → final.mem q off=s.mem q off) := by
  obtain ⟨t,ht,he,_,_⟩ := loop_run hoist M equal32 s X W Bias Y Mean Rstd S D B 0
    (blockCount D B) row tree mean inverse hB (by omega) hX hW hBias hm hr hx indexPins pins
  refine ⟨t,by simpa using ht,he.registers,?_,?_⟩
  · intro col hc
    obtain ⟨hk,hi,heq⟩ := column_chunk D B col hB hc
    have hv := he.values (col/B) (Nat.zero_le _) (by simpa using hk) ⟨col%B,hi⟩ (by simpa only [heq] using hc)
    simpa only [heq] using hv
  · intro q off hmiss
    apply he.frame
    rcases hmiss with hq | ho
    · exact Or.inl hq
    · exact Or.inr (fun k _ _ i hi => ho (k*B+i.val) hi)

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableNormalizeLoop
