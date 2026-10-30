import bench.repairs.tritonbench_g.attention_llama.Proofs.FP8Loop
import bench.tritonbench_g.attention_llama.Proofs.FP8RealSetup
import bench.tritonbench_g.attention_llama.Proofs.FP8RealPostamble

namespace VeriTile.Bench.Repairs.AttentionLlama.FP8Execution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open RealKernel FP8RealMemory FP8Loop
open FP8RealLoop (BytesFrom)
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

private theorem inline_loop (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) (s : BlockState) :
    stepStmts (SourceExecution.sourceLoop scale skn svk N SP BM BN BD causal) s =
      stepStmts (FP8Iteration.referenceLoop scale skn svk N SP BM BN BD causal) s := by
  simp only [SourceExecution.sourceLoop, FP8Iteration.referenceLoop, ProgramRewrite.append,
    ProgramRewrite.constant_branch,SourceExecution.sourceTail_step,Option.bind_assoc]

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)

def referenceBody : List Stmt :=
  alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
    .ifThen (.constBool causal) [alBnEndCausalStmt SP BN] ::
    .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
      (FP8Iteration.referenceLoop scale skn svk N SP BM BN BD causal) ::
      alPostLoopG Out soz soh som son H BM BD

theorem reference_execution (s : BlockState) :
    exec (TypedSource.repaired Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel s =
      stepStmts (referenceBody Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal) s := by
  simp only [exec,SourceExecution.repaired_body,referenceBody,ProgramRewrite.append,stepStmts,
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _ (inline_loop scale skn svk N SP BM BN BD causal)]

/-- Real erasure of the complete raw-byte source agrees with the real source
on decoded inputs. Shapes, strides and both source flags remain symbolic.
Finite-byte checks apply only to the actual reached and unmasked K/V loads. -/
theorem decoded_execution (s : BlockState)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hundef : ∀ r o, s.undef r o = 0)
    (hK : BytesFrom K s (alBase s NH skz skh) skn skk 0
      (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (hV : BytesFrom V s (alBase s NH svz svh) svk svn 0
      (FP8RealSetup.stop s N SP BN causal) N BN BD) :
    (exec (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel s).map
        (decoded K V) =
    exec (TypedSource.repaired Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel
        (decoded K V s) := by
  obtain ⟨entry, hentry, hc, hstop⟩ := FP8RealSetup.runs Q K V sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N SP BM BN BD causal s hundef
  have hview := FP8RealSetup.decoded_execution Q K V sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N SP BM BN BD causal s hQK hQV
  rw [hentry, Option.map_some] at hview
  have hloop := dynamic_range K V s entry scale (alBase s NH skz skh) (alBase s NH svz svh)
    skn skk svk svn (FP8RealSetup.stop s N SP BN causal) N SP BM BN BD causal hc hstop hK hV
  rw [FP8Source.fused_execution, reference_execution]
  change (stepStmts (FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true) s).map _ = _
  have hbody : FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true =
      FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD causal ++
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) ::
        alPostLoopG Out soz soh som son H BM BD := by
    simp [FP8Source.body, FP8RealSetup.body, List.append_assoc]
  have href : referenceBody Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal =
      FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD causal ++
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8Iteration.referenceLoop scale skn svk N SP BM BN BD causal) ::
        alPostLoopG Out soz soh som son H BM BD := by
    simp [referenceBody, FP8RealSetup.body, List.append_assoc]
  rw [hbody, href, stepStmts.append_some hentry, stepStmts.append_some hview.symm]
  simp only [stepStmts]
  rw [← hloop]
  cases he : stepStmt (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end")
      (.constNat BN) (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal)) entry with
  | none => simp
  | some mid =>
    simp only [Option.map_some]
    exact FP8RealPostamble.postamble K V Out mid soz soh som son H BM BD hOK hOV

/-- Successful source execution preserves every raw cell outside Out. This
is stronger than preservation of decoded values and keeps original bytes. -/
theorem memory_frame (s t : BlockState) (r : RegionName) (o : Nat)
    (hr : r ≠ Out) (hundef : ∀ r o, s.undef r o = 0)
    (hrun : exec (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel s = some t) :
    t.mem r o = s.mem r o := by
  obtain ⟨entry, hentry, hc, hstop⟩ := FP8RealSetup.runs Q K V sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N SP BM BN BD causal s hundef
  rw [FP8Source.fused_execution] at hrun
  change stepStmts (FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true) s = some t at hrun
  have hbody : FP8Source.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true =
      FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD causal ++
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) ::
        alPostLoopG Out soz soh som son H BM BD := by
    simp [FP8Source.body, FP8RealSetup.body, List.append_assoc]
  rw [hbody, stepStmts.append_some hentry] at hrun
  simp only [stepStmts, stepForRangeAux.forRangeDyn_unfold, evalOp_constNat,
    evalOp_ref, hstop, Option.bind_some, Tile.scalar_data] at hrun
  cases he : stepForRangeAux "start_n" 0 (FP8RealSetup.stop s N SP BN causal) BN
      (FP8Source.loopBody Bool.true scale skn svk N SP BM BN BD causal) entry with
  | none => simp [he] at hrun
  | some mid =>
    simp only [he] at hrun
    have hm := range_memory entry mid scale skn svk 0 (FP8RealSetup.stop s N SP BN causal)
      N SP BM BN BD causal he
    exact (FP8RealPostamble.memory_frame Out r o mid t soz soh som son H BM BD hr hrun).trans
      (congrFun (congrFun (hm.trans hc.memory) r) o)

end
end VeriTile.Bench.Repairs.AttentionLlama.FP8Execution
