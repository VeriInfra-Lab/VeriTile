import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeDiagonalReads

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReads
open VeriTile Triton FP.Structural
open FPStorage (Memory state)
open FPStages (Stage)
open NativeLaunch (start)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem output_feature (M : Algebra α) (c : OutputGrid.Shape) (scale : ℝ)
    (idx : GridIndex (OutputGrid.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.output c scale) s m) (r : RegionName) (hr : r="q" ∨ r="g") (f : Nat) :
    FPOutput.featureLoad M c (start (.output c scale) idx s) r f=
      FPOutput.featureLoad M c (state m idx.toPids (OutputGrid.grid c).toNumPids) r f := by
  funext i
  have port : (r,c.heads*c.T*c.K)∈NativeLaunch.inputExtents (.output c scale) := by
    rcases hr with rfl|rfl <;> simp [NativeLaunch.inputExtents]
  have rh : r≠"h" := by rcases hr with rfl|rfl <;> decide
  dsimp only [FPOutput.featureLoad,FPOutput.row,start,state,Stage.grid]
  by_cases active : FPOutput.row c (start (.output c scale) idx s) i.1.val<c.T ∧
      FPOutput.feature c f i.2.1.val<c.K
  · have hb := FPReadBounds.output_features c idx m f i active.1 active.2
    dsimp only [FPOutput.row,start,state,Stage.grid] at active hb
    rw [if_pos active,if_pos active]
    apply congrArg M.fp32Load
    exact (real_read reads r _ port rh _ hb).trans (FPStorage.import_real _ _ _ rh).symm
  · dsimp only [FPOutput.row,start,Stage.grid] at active
    rw [if_neg active,if_neg active]

theorem output_history (M : Algebra α) (c : OutputGrid.Shape) (scale : ℝ)
    (idx : GridIndex (OutputGrid.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.output c scale) s m) (f : Nat) :
    FPOutput.historyLoad M c (start (.output c scale) idx s) f=
      FPOutput.historyLoad M c (state m idx.toPids (OutputGrid.grid c).toNumPids) f := by
  funext i
  dsimp only [FPOutput.historyLoad,FPOutput.column,start,state,Stage.grid]
  by_cases active : FPOutput.feature c f i.1.val<c.K ∧
      FPOutput.column c (start (.output c scale) idx s) i.2.1.val<c.V
  · have hb := FPReadBounds.output_history c idx m f i active.1 active.2
    dsimp only [FPOutput.column,start,state,Stage.grid] at active hb
    rw [if_pos active,if_pos active]
    apply congrArg (M.cast (some .fp32) .fp32 .real)
    exact (history_read reads _ (by simp [NativeLaunch.inputExtents]) _ hb).trans
      (FPStorage.import_history _ _).symm
  · dsimp only [FPOutput.column,start,Stage.grid] at active
    rw [if_neg active,if_neg active]

theorem output_attention (M : Algebra α) (c : OutputGrid.Shape) (scale : ℝ)
    (idx : GridIndex (OutputGrid.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.output c scale) s m) :
    FPOutput.attentionLoad M c (start (.output c scale) idx s)=
      FPOutput.attentionLoad M c (state m idx.toPids (OutputGrid.grid c).toNumPids) := by
  funext i
  dsimp only [FPOutput.attentionLoad,FPOutput.row,start,state,Stage.grid]
  by_cases causal : i.2.1.val ≤ i.1.val
  · rw [if_pos causal,if_pos causal]
    by_cases active : FPOutput.row c (start (.output c scale) idx s) i.1.val<c.T
    · have hb := FPReadBounds.output_attention c idx m i active
      dsimp only [FPOutput.row,start,state,Stage.grid] at active hb
      rw [if_pos active,if_pos active]
      apply congrArg (M.cast (some .fp32) .real .real)
      apply congrArg M.fp32Load
      exact (real_read reads "attention" _ (by simp [NativeLaunch.inputExtents]) (by decide) _ hb).trans
        (FPStorage.import_real _ _ _ (by decide)).symm
    · dsimp only [FPOutput.row,start,Stage.grid] at active
      rw [if_neg active,if_neg active]
  · rw [if_neg causal,if_neg causal]

theorem output_load_value (M : Algebra α) (c : OutputGrid.Shape) (scale : ℝ)
    (idx : GridIndex (OutputGrid.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.output c scale) s m) :
    FPOutput.valueLoad M c (start (.output c scale) idx s)=
      FPOutput.valueLoad M c (state m idx.toPids (OutputGrid.grid c).toNumPids) := by
  funext i
  dsimp only [FPOutput.valueLoad,FPOutput.active,FPOutput.outputAddress,
    FPOutput.row,FPOutput.column,start,state,Stage.grid]
  by_cases active : FPOutput.active c (start (.output c scale) idx s) i
  · have hb := FPStages.output_bound c idx m i active
    dsimp only [FPOutput.active,FPOutput.outputAddress,FPOutput.row,FPOutput.column,
      start,state,Stage.grid] at active hb
    rw [if_pos active,if_pos active]
    apply congrArg M.fp32Load
    exact (real_read reads "v" _ (by simp [NativeLaunch.inputExtents]) (by decide) _ hb).trans
      (FPStorage.import_real _ _ _ (by decide)).symm
  · dsimp only [FPOutput.active,FPOutput.row,FPOutput.column,start,Stage.grid] at active
    rw [if_neg active,if_neg active]

theorem output_value (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : OutputGrid.Shape) (scale : ℝ) (idx : GridIndex (OutputGrid.grid c))
    (s : State α) (m : Memory α) (reads : Reads (.output c scale) s m) :
    FPOutput.value M D c scale (start (.output c scale) idx s)=
      FPOutput.value M D c scale (state m idx.toPids (OutputGrid.grid c).toNumPids) := by
  have query (f : Nat) : FPOutput.queryGate M c scale (start (.output c scale) idx s) f=
      FPOutput.queryGate M c scale (state m idx.toPids (OutputGrid.grid c).toNumPids) f := by
    funext i
    simp only [FPOutput.queryGate,output_feature M c scale idx s m reads "q" (Or.inl rfl),
      output_feature M c scale idx s m reads "g" (Or.inr rfl)]
  have update : FPOutput.update M D c scale (start (.output c scale) idx s)=
      FPOutput.update M D c scale (state m idx.toPids (OutputGrid.grid c).toNumPids) := by
    funext f a i
    simp only [FPOutput.update,query,output_history M c scale idx s m reads]
  funext i
  simp only [FPOutput.value,FPOutput.accumulated,update,output_attention M c scale idx s m reads,
    output_load_value M c scale idx s m reads]

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReads
