import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealSetup

/-! Actual byte-load address safety through the complete streaming loop.
Arithmetic registers may be arbitrary, and decode failures are allowed. Only
visited block starts and unmasked lanes impose allocation requirements. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafetyLoop
open VeriTile.Triton RealKernel FP8RealIteration FP8RealLoop FP8RealLoopFrame
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

private theorem max_head_safe {n : Nat} (bounds : RegionBounds) (x y : Op .real [n])
    (name : RegName) (rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.assign .real [n] name (.max2 (.consSame .nil) x y) :: rest) s ↔
      Stmt.TraceSafeList bounds (.assign .real [n] name (.maximumFloat .real (.consSame .nil) x y) :: rest) s := by
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, stepStmt, FloatDType.toTileDType]
  erw [evalOp_maximumFloat_real]
  conv_lhs => arg 1; rw [Op.SafeAt.eq_def]
  conv_rhs => arg 1; rw [Op.SafeAt.eq_def]
  rfl

private theorem source_tail_safe (bounds : RegionBounds) (skn svk N BM BN BD : Nat)
    (candidate : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds
      ((FP8RealSource.sourceTail skn svk N BM BN BD candidate).flatMap (FP8RealSource.lowerLoad Bool.true)) s ↔
    Stmt.TraceSafeList bounds
      ((alLoopTailG skn svk N BM BN BD candidate).flatMap (FP8RealSource.lowerLoad Bool.true)) s := by
  exact max_head_safe bounds _ _ "m_curr"
    (((alLoopTailG skn svk N BM BN BD candidate).drop 1).flatMap (FP8RealSource.lowerLoad Bool.true)) s

private theorem load_safe (bounds : RegionBounds) (R : RegionName) (s : BlockState)
    (name : RegName) (base sr sc SN N BN BD : Nat)
    (hptr : s.regs .ptr [BN, BD] name = some (alPtrTile R base sr sc SN BN BD))
    (hrows : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hb : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      base + (SN + i.1.val) * sr + i.2.1.val * sc < bounds R) :
    Op.SafeAt bounds s (.decodeFloat8E5 .real
      (.load .int (.ptr (.ref .ptr [BN, BD] name))
        (.maskOther (keyMask N BN BD) (.broadcast (.constInt 0) [BN, BD])))) := by
  simp only [Op.SafeAt.eq_def, keyMask, true_and, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def,
    MaskOpt.Active]
  intro ptrs hp i hi
  rw [evalOp_ref, hptr] at hp
  cases hp
  rcases hi with ⟨mask, hm, hi⟩
  rw [al_colmask_eval s SN N hrows] at hm
  cases hm
  exact hb i (of_decide_eq_true hi)

/-- Address bounds follow the actual range counter and the source's load mask. -/
def AccessesFrom (bounds : RegionBounds) (R : RegionName)
    (base sr sc cur stop N BN BD : Nat) : Prop :=
  ∀ k : Nat, cur + k * BN < stop → ∀ (j : Fin BN) (d : Fin BD),
    cur + k * BN + j.val < N →
      base + (cur + k * BN + j.val) * sr + d.val * sc < bounds R

private theorem accesses_next (bounds : RegionBounds) (R : RegionName)
    (base sr sc cur stop N BN BD : Nat)
    (h : AccessesFrom bounds R base sr sc cur stop N BN BD) :
    AccessesFrom bounds R base sr sc (cur + BN) stop N BN BD := by
  intro k hk j d hj
  have he : cur + BN + k * BN = cur + (k + 1) * BN := by
    simp only [Nat.add_mul, Nat.one_mul]
    omega
  rw [he] at hk hj ⊢
  exact h (k + 1) hk j d hj

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafetyLoop
