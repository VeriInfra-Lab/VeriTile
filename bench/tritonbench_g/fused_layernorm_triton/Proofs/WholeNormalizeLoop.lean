import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeNormalize
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeGeometry

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def normalizeCode (X W Bias Y : RegionName) (D B : Nat) : Nat → Nat → List ComputeStmt
  | _, 0 => []
  | start, count + 1 => normalizeChunk X W Bias Y D B start ++
      normalizeCode X W Bias Y D B (start + 1) count

structure NormalizeEffect {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X W Bias Y : RegionName) (D B start count row : Nat) (mean inverse : α) : Prop where
  registers : NormalizeRegisters s t row mean inverse
  values : ∀ k, start ≤ k → k < start + count → ∀ i : Fin B, k * B + i.val < D →
    t.mem Y (k * B + i.val + D * row) =
      .mk .bf16 (normalizeValue M s X W Bias D row (k * B + i.val) mean inverse)
  frame : ∀ q off, (q ≠ Y ∨ ∀ k, start ≤ k → k < start + count →
    ∀ i : Fin B, k * B + i.val < D → off ≠ k * B + i.val + D * row) →
      t.mem q off = s.mem q off

theorem normalize_chunk_effect {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (D B k row : Nat) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) (normalizeChunk X W Bias Y D B k) s = some final ∧
      NormalizeEffect M s final X W Bias Y D B k 1 row mean inverse := by
  obtain ⟨t, ht, hmem, hreg⟩ := normalize_chunk_run M equal32 s X W Bias Y D B k row mean inverse hm hr hx
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
    (D B start count row : Nat) (mean inverse : α)
    (hX : X ≠ Y) (hW : W ≠ Y) (hBias : Bias ≠ Y)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) (normalizeCode X W Bias Y D B start count) s = some final ∧
      NormalizeEffect M s final X W Bias Y D B start count row mean inverse := by
  induction count generalizing start s with
  | zero =>
    refine ⟨s, by simp [normalizeCode, run], ⟨rfl, rfl, hm, hr, hx⟩, ?_, ?_⟩
    · intro k hl hu; omega
    · intro q off h; rfl
  | succ count ih =>
    obtain ⟨u, hu, he⟩ := normalize_chunk_effect M equal32 s X W Bias Y D B start row mean inverse hm hr hx
    obtain ⟨t, ht, hf⟩ := ih u (start + 1) he.registers.mean he.registers.rstd he.registers.row
    have hXu : u.mem X = s.mem X := funext (fun off => he.frame X off (.inl hX))
    have hWu : u.mem W = s.mem W := funext (fun off => he.frame W off (.inl hW))
    have hBu : u.mem Bias = s.mem Bias := funext (fun off => he.frame Bias off (.inl hBias))
    refine ⟨t, by simp [normalizeCode, run_append, hu, ht], ?_⟩
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
    (D B row : Nat) (mean inverse : α) (hB : 0 < B)
    (hX : X ≠ Y) (hW : W ≠ Y) (hBias : Bias ≠ Y)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32) (normalizeCode X W Bias Y D B 0 (blockCount D B)) s = some final ∧
      NormalizeRegisters s final row mean inverse ∧
      (∀ col, col < D → final.mem Y (col + D * row) =
        .mk .bf16 (normalizeValue M s X W Bias D row col mean inverse)) ∧
      (∀ q off, (q ≠ Y ∨ ∀ col, col < D → off ≠ col + D * row) → final.mem q off = s.mem q off) := by
  obtain ⟨t, ht, he⟩ := normalize_loop_run M equal32 s X W Bias Y D B 0 (blockCount D B) row mean inverse hX hW hBias hm hr hx
  refine ⟨t, ht, he.registers, ?_, ?_⟩
  · intro col hc
    obtain ⟨hk, hi, heq⟩ := column_chunk D B col hB hc
    have hv := he.values (col / B) (Nat.zero_le _) (by simpa using hk) ⟨col % B, hi⟩ (by simpa only [heq] using hc)
    simpa only [heq] using hv
  · intro q off hmiss
    apply he.frame
    rcases hmiss with hq | ho
    · exact Or.inl hq
    · exact Or.inr (fun k _ _ i hi => ho (k * B + i.val) hi)

#axiomsClean normalize_loop_run
#axiomsClean normalize_whole_row
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
