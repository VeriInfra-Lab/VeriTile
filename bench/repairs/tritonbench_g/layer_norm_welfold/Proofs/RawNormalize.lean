import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawStatistics
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Raw
open VeriTile.Triton FP.Structural Reuse
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The original source keeps a two-dimensional tile through its final store. -/
def outputChunk {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias Y : RegionName) (D B k row : Nat) (mean inverse : α) : State α :=
  (TileShape.allIndices [1,B]).foldl (fun t i =>
    if k*B+i.2.1.val<D then
      t.write Y (k*B+i.2.1.val+D*row)
        (.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k*B+i.2.1.val) mean inverse))
    else t) s

structure NormalizeRegisters {α : Type} (s t : State α) (B row : Nat)
    (mean inverse : α) : Prop where
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  rstd : t.regs .real [1,1] "tmp18"=some (fun _ => inverse)
  row : t.regs .nat [1,1] "x0"=some (fun _ => row)
  base : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)

structure NormalizeEffect {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X W Bias Y : RegionName) (D B start count row : Nat) (mean inverse : α) : Prop where
  registers : NormalizeRegisters s t B row mean inverse
  values : ∀ k, start≤k → k<start+count → ∀ i : Fin B, k*B+i.val<D →
    t.mem Y (k*B+i.val+D*row)=.mk .bf16
      (normalizeValue M s X W Bias D row (k*B+i.val) mean inverse)
  frame : ∀ q off, (q≠Y ∨ ∀ k, start≤k → k<start+count →
    ∀ i : Fin B, k*B+i.val<D → off≠k*B+i.val+D*row) → t.mem q off=s.mem q off

theorem normalize_chunk_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B k row : Nat) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp6" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp18" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row))
    (hb : s.regs .nat [1,B] "rbase" = some (fun i => i.2.1.val)) :
    ∃ final, run (withEquality M equal32) (normalizeBody reuse X W Bias Y S D B (Readable.tree rt))
        (s.setReg "roffset" .nat [] (fun _ => k*B)) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final B row mean inverse := by
  cases reuse <;>
    simp [normalizeBody,SourceCode.selected,SourceCode.rowKernel,SourceCode.rowOffsetReusedKernel,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.numericLt,withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,hb,store,
      maskedNormalizeValue,outputChunk]
  all_goals constructor
  all_goals first
    | (apply typed_scatter_memory_congr; simp only [State.setReg_mem])
    | (constructor <;> simp [typed_scatter_regs,typed_scatter_undef,hm,hr,hx,hb])

theorem normalize_chunk_effect {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B k row : Nat) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp6" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp18" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row))
    (hb : s.regs .nat [1,B] "rbase" = some (fun i => i.2.1.val)) :
    ∃ final, run (withEquality M equal32) (normalizeBody reuse X W Bias Y S D B (Readable.tree rt)) (s.setReg "roffset" .nat [] (fun _ => k*B)) = some final ∧
      NormalizeEffect M s final X W Bias Y D B k 1 row mean inverse := by
  obtain ⟨t, ht, hmem, hreg⟩ := normalize_chunk_run M equal32 s X W Bias Y S D B k row rt reuse mean inverse hm hr hx hb
  refine ⟨t, ht, hreg, ?_, ?_⟩
  · intro l hl hu i hi
    have heq : l = k := by omega
    subst l
    rw [hmem]
    have hv := State.masked_scatter_readback Y
      (fun i : TileIndex [1,B] => k * B + i.2.1.val + D * row)
      (fun i => Cell.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k * B + i.2.1.val) mean inverse))
      (fun i => k * B + i.2.1.val < D) s (⟨0,by decide⟩,i, ()) hi
      (by intro a b hab; dsimp at hab; exact Prod.ext (Subsingleton.elim _ _) (Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)))
    simpa [outputChunk, maskedNormalizeValue, normalizeValue, hi] using hv
  · intro q off hmiss
    rw [hmem]
    apply State.masked_scatter_frame
    rcases hmiss with hq | ho
    · exact Or.inl hq
    · exact Or.inr (fun i _ hi => ho k (by omega) (by omega) i.2.1 hi)

theorem normalize_loop_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (S D B start count row : Nat) (hB : 0<B)
    (count_eq : start+count=blockCount D B) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hX : X ≠ Y) (hW : W ≠ Y) (hBias : Bias ≠ Y)
    (hm : s.regs .real [1,1] "tmp6" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp18" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row))
    (hb : s.regs .nat [1,B] "rbase" = some (fun i => i.2.1.val)) :
    ∃ final, FP.Structural.range (withEquality M equal32) "roffset" (start*B) D B (normalizeBody reuse X W Bias Y S D B (Readable.tree rt)) s = some final ∧
      NormalizeEffect M s final X W Bias Y D B start count row mean inverse := by
  induction count generalizing start s with
  | zero =>
    have stopped : ¬start*B<D := by
      intro active
      have := (range_offset_iff D B start hB).mpr active
      omega
    refine ⟨s, by rw [range,if_neg (Nat.ne_of_gt hB),if_neg stopped], ⟨rfl, rfl, hm, hr, hx, hb⟩, ?_, ?_⟩
    · intro k hl hu; omega
    · intro q off h; rfl
  | succ count ih =>
    obtain ⟨u, hu, he⟩ := normalize_chunk_effect M equal32 s X W Bias Y S D B start row rt reuse mean inverse hm hr hx hb
    obtain ⟨t, ht, hf⟩ := ih u (start + 1) (by omega) he.registers.mean he.registers.rstd he.registers.row he.registers.base
    have hXu : u.mem X = s.mem X := funext (fun off => he.frame X off (.inl hX))
    have hWu : u.mem W = s.mem W := funext (fun off => he.frame W off (.inl hW))
    have hBu : u.mem Bias = s.mem Bias := funext (fun off => he.frame Bias off (.inl hBias))
    refine ⟨t, ?_, ?_⟩
    · have active : start*B<D := (range_offset_iff D B start hB).mp (by omega)
      rw [range,if_neg (Nat.ne_of_gt hB),if_pos active,hu]
      simpa only [Option.bind_some,show start*B+B=(start+1)*B by simp [Nat.add_mul]] using ht
    refine ⟨⟨hf.registers.pids.trans he.registers.pids,
      hf.registers.undef.trans he.registers.undef, hf.registers.mean,
      hf.registers.rstd, hf.registers.row,hf.registers.base⟩, ?_, ?_⟩
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
    (hm : s.regs .real [1,1] "tmp6" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp18" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row))
    (hb : s.regs .nat [1,B] "rbase" = some (fun i => i.2.1.val)) :
    ∃ final, run (withEquality M equal32) [.forRange "roffset" 0 D B (normalizeBody reuse X W Bias Y S D B (Readable.tree rt))] s = some final ∧
      NormalizeRegisters s final B row mean inverse ∧
      (∀ col, col < D → final.mem Y (col + D * row) =
        .mk .bf16 (normalizeValue M s X W Bias D row col mean inverse)) ∧
      (∀ q off, (q ≠ Y ∨ ∀ col, col < D → off ≠ col + D * row) → final.mem q off = s.mem q off) := by
  obtain ⟨t, ht, he⟩ := normalize_loop_run M equal32 s X W Bias Y S D B 0 (blockCount D B) row hB (by simp) rt reuse mean inverse hX hW hBias hm hr hx hb
  simp only [Nat.zero_mul] at ht
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
end VeriTile.Bench.Repairs.LayerNormWelfold.Raw
