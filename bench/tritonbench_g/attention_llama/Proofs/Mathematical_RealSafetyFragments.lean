import bench.tritonbench_g.attention_llama.Proofs.Mathematical_RealKernel
import VeriTile.Triton.Memory.TraceProofs

/-! Safety at actual masked pointer accesses. Arithmetic registers remain
abstract: these bounds impose no attention formula or FP assumption. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafetyFragments
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The V access is protected by the same global row mask. Either l_curr
operand order is safe; none of its numerical values is assumed. -/
theorem tail_safe (bounds : RegionBounds) (skn svk N BM BN BD SN : Nat)
    (s : BlockState) (V : RegionName) (base svn : Nat) (candidate : Bool)
    (hoff : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hptr : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V base svk svn SN BN BD))
    (hb : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      base + (SN + i.1.val) * svk + i.2.1.val * svn < bounds V) :
    Stmt.TraceSafeList bounds (alLoopTailG skn svk N BM BN BD candidate) s := by
  cases candidate
  all_goals
    simp only [alLoopTailG, Bool.false_eq_true, ↓reduceIte]
    -- The eight numeric assignments before v do not alter its pointer or mask registers.
    iterate 8
      refine TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def]) ?_
      intro value
    apply TraceProofs.assign_cons
    · simp only [Op.SafeAt.eq_def, true_and]
      simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def, true_and, MaskOpt.Active]
      intro ptrs hp i hi
      simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, hptr] at hp
      cases hp
      rcases hi with ⟨mask, hm, hi⟩
      rw [al_colmask_eval _ SN N (by
        simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] using hoff)] at hm
      cases hm
      exact hb i (of_decide_eq_true hi)
    · intro _
      apply TraceProofs.all_safe
      simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafetyFragments
