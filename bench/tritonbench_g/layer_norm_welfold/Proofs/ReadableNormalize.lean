import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableStatistics
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem normalize_chunk_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B k row : Nat) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) (normalizeBody reuse X W Bias Y S D B rt)
        (s.setReg "k" .nat [] (fun _ => k)) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse := by
  cases reuse <;>
    simp [normalizeBody,SourceCode.selected,SourceCode.rowKernel,SourceCode.rowOffsetReusedKernel,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.numericLt,withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,store,
      maskedNormalizeValue,outputChunk]
  all_goals constructor
  all_goals first
    | (apply typed_scatter_memory_congr; simp only [State.setReg_mem])
    | (constructor <;> simp [typed_scatter_regs,typed_scatter_undef,hm,hr,hx])

theorem normalize_chunk_effect {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B k row : Nat) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) (normalizeBody reuse X W Bias Y S D B rt) (s.setReg "k" .nat [] (fun _ => k)) = some final ∧
      NormalizeEffect M s final X W Bias Y D B k 1 row mean inverse := by
  obtain ⟨t, ht, hmem, hreg⟩ := normalize_chunk_run M equal32 s X W Bias Y S D B k row rt reuse mean inverse hm hr hx
  refine ⟨t, ht, hreg, ?_, ?_⟩
  · intro l hl hu i hi
    have heq : l = k := by omega
    subst l
    rw [hmem]
    have hv := State.masked_scatter_readback Y
      (fun i : TileIndex [B] => k * B + i.1.val + D * row)
      (fun i => Cell.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k * B + i.1.val) mean inverse))
      (fun i => k * B + i.1.val < D) s (i, ()) hi
      (by intro a b hab; dsimp at hab; exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _))
    simpa [outputChunk, maskedNormalizeValue, normalizeValue, hi] using hv
  · intro q off hmiss
    rw [hmem]
    apply State.masked_scatter_frame
    rcases hmiss with hq | ho
    · exact Or.inl hq
    · exact Or.inr (fun i _ hi => ho k (by omega) (by omega) i.1 hi)

theorem normalize_loop_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B start count row : Nat) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hX : X ≠ Y) (hW : W ≠ Y) (hBias : Bias ≠ Y)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, FP.Structural.loop (withEquality M equal32) "k" start (start+count) (normalizeBody reuse X W Bias Y S D B rt) s = some final ∧
      NormalizeEffect M s final X W Bias Y D B start count row mean inverse := by
  induction count generalizing start s with
  | zero =>
    refine ⟨s, by simp [FP.Structural.loop], ⟨rfl, rfl, hm, hr, hx⟩, ?_, ?_⟩
    · intro k hl hu; omega
    · intro q off h; rfl
  | succ count ih =>
    obtain ⟨u, hu, he⟩ := normalize_chunk_effect M equal32 s X W Bias Y S D B start row rt reuse mean inverse hm hr hx
    obtain ⟨t, ht, hf⟩ := ih u (start + 1) he.registers.mean he.registers.rstd he.registers.row
    have hXu : u.mem X = s.mem X := funext (fun off => he.frame X off (.inl hX))
    have hWu : u.mem W = s.mem W := funext (fun off => he.frame W off (.inl hW))
    have hBu : u.mem Bias = s.mem Bias := funext (fun off => he.frame Bias off (.inl hBias))
    refine ⟨t, ?_, ?_⟩
    · rw [FP.Structural.loop,if_pos (by omega),hu]
      simpa only [Option.bind,show start+Nat.succ count=(start+1)+count by omega] using ht
    refine ⟨⟨hf.registers.pids.trans he.registers.pids,
      hf.registers.undef.trans he.registers.undef, hf.registers.mean,
      hf.registers.rstd, hf.registers.row⟩, ?_, ?_⟩
    · intro k hl hupp i hi
      by_cases hk : k = start
      · subst k
        rw [hf.frame Y (start * B + i.val + D * row) ?_]
        · exact he.values start (by omega) (by omega) i hi
        · right
          intro l hls hlu j hj heq
          have hs := (chunk_unique B start l i.val j.val i.isLt j.isLt (by omega)).1
          omega
      · have hv := hf.values k (by omega) (by omega) i hi
        simpa only [normalizeValue, hXu, hWu, hBu] using hv
    · intro q off hmiss
      rw [hf.frame q off ?_, he.frame q off ?_]
      · rcases hmiss with hq | ho
        · exact Or.inl hq
        · exact Or.inr (fun k hl hu i hi => ho k (by omega) (by omega) i hi)
      · rcases hmiss with hq | ho
        · exact Or.inl hq
        · exact Or.inr (fun k hl hu i hi => ho k (by omega) (by omega) i hi)

/-- Every logical output column is written in BF16, and every other cell is framed. -/
theorem normalize_whole_row {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B row : Nat) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α) (hB : 0 < B)
    (hX : X ≠ Y) (hW : W ≠ Y) (hBias : Bias ≠ Y)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) [.forLoop "k" (blockCount D B) (normalizeBody reuse X W Bias Y S D B rt)] s = some final ∧
      NormalizeRegisters s final row mean inverse ∧
      (∀ col, col < D → final.mem Y (col + D * row) =
        .mk .bf16 (normalizeValue M s X W Bias D row col mean inverse)) ∧
      (∀ q off, (q ≠ Y ∨ ∀ col, col < D → off ≠ col + D * row) → final.mem q off = s.mem q off) := by
  obtain ⟨t, ht, he⟩ := normalize_loop_run M equal32 s X W Bias Y S D B 0 (blockCount D B) row rt reuse mean inverse hX hW hBias hm hr hx
  simp only [Nat.zero_add] at ht
  refine ⟨t, by simp [run,step,ht], he.registers, ?_, ?_⟩
  · intro col hc
    obtain ⟨hk, hi, heq⟩ := column_chunk D B col hB hc
    have hv := he.values (col / B) (Nat.zero_le _) (by simpa using hk) ⟨col % B, hi⟩ (by simpa only [heq] using hc)
    simpa only [heq] using hv
  · intro q off hmiss
    apply he.frame
    rcases hmiss with hq | ho
    · exact Or.inl hq
    · exact Or.inr (fun k _ _ i hi => ho (k * B + i.val) hi)


#axiomsClean normalize_whole_row
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
