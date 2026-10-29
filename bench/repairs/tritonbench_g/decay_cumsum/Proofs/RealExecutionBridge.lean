import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealPrefixBridge
import bench.tritonbench_g.decay_cumsum.Proofs.PrepareExecution

namespace VeriTile.Bench.Repairs.DecayCumsum.RealSourceBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DecayCumsum
open RealBodyBridge
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def legacyLoop (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : Stmt :=
  match kind with
  | .forward | .prepare => .forRangeDyn "i" (.constNat 0) (.constNat p.span)
      (.constNat 1) (RealSourceBridge.body kind h p scale)
  | .backward => .forRangeInt "__rev_j"
      (.sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (p.span : Int)) (.constInt 1)))
      (.constInt 1) (.constNat 1) (RealSourceBridge.body kind h p scale)

theorem legacy_source_body (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (LegacySurface.source kind h p scale).eraseDType.toAlgKernel.body =
      RealPrefixBridge.legacyPrefix kind h p scale ++ [legacyLoop kind h p scale] := by
  cases kind
  all_goals simp only [LegacySurface.source,Original.forward,Original.prepare,Original.backward,
    TypedSource.forward,TypedSource.prepare,TypedSource.backward,TypedSource.inv_ln2,
    RealPrefixBridge.legacyPrefix,prefixCount,legacyLoop,body,RealReference.fwdBody,RealReference.prepBody,
    RealBodyBridge.sourceBody,RealReference.bwdIterBodyG,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  all_goals rfl

theorem body_flat (A : FlatAlloc) (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    A.flattenStmts (RealSourceBridge.body kind h p scale)=body kind h p scale := by
  cases kind <;> simp only [RealSourceBridge.body,RealBodyBridge.sourceBody,RealReference.fwdBody,
    RealReference.prepBody,RealReference.bwdIterBodyG,List.drop,List.cons_append,List.nil_append,
    ComputeDType.eraseDType,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask,Op.broadcast]

theorem loop_flat (A : FlatAlloc) (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    A.flattenStmts [loop kind h p scale]=[loop kind h p scale] ∧
    A.flattenStmts [legacyLoop kind h p scale]=[legacyLoop kind h p scale] := by
  cases kind <;> simp only [RealSourceBridge.loop,legacyLoop,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp,body_flat] <;> trivial

theorem loop_equivalent (kind : Kind) (h : Call) (p : Phase) (scale : ℝ)
    (s t : BlockState) (he : RealBodyBridge.norm s=RealBodyBridge.norm t) :
    (stepStmts [legacyLoop kind h p scale] s).map RealBodyBridge.norm =
      (stepStmts [loop kind h p scale] t).map RealBodyBridge.norm := by
  rw [PrepareExecution.singleton,PrepareExecution.singleton]
  cases kind
  all_goals simp only [legacyLoop,RealSourceBridge.loop,stepStmt,evalOp,Tile.uop,Tile.scalar_data,
    Tile.bop,NumericDType.sub,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some,bind]
  · change (stepForRangeAux "i" 0 p.span 1 (RealSourceBridge.body .forward h p scale) s).map RealBodyBridge.norm =
      (stepForRangeIntAux "i" 0 (p.span : Int) 1 (RealSourceBridge.body .forward h p scale) t).map RealBodyBridge.norm
    exact unsigned_signed_range _ (forward_transports "g" "go" h.headStride p.span h.BK h.width) 0 p.span s t he
  · change (stepForRangeAux "i" 0 p.span 1 (RealSourceBridge.body .prepare h p scale) s).map RealBodyBridge.norm =
      (stepForRangeIntAux "i" 0 (p.span : Int) 1 (RealSourceBridge.body .prepare h p scale) t).map RealBodyBridge.norm
    exact unsigned_signed_range _ (prepare_transports "q" "k" "g" "qg" "kg" h.headStride h.width p.span h.BK scale) 0 p.span s t he
  · change (stepForRangeIntAux "__rev_j" (0-((p.span : Int)-1)) 1 1 (RealSourceBridge.body .backward h p scale) s).map RealBodyBridge.norm =
      (stepForRangeIntAux "__rev_j" (0-((p.span : Int)-1)) 1 1 (RealSourceBridge.body .backward h p scale) t).map RealBodyBridge.norm
    have left := RealBodyBridge.range h.width p.span h.BK 0 s s rfl
    have right := RealBodyBridge.range h.width p.span h.BK 0 s t he
    have bound : (0:Int)-((p.span : Int)-1)=1-(p.span : Int)+0 := by ring
    simpa only [RealSourceBridge.body,bound] using left.symm.trans right

theorem sequence (lp rp lc rc : List Stmt) (u : BlockState)
    (preludes : (stepStmts lp u).map RealBodyBridge.norm=(stepStmts rp u).map RealBodyBridge.norm)
    (loops : ∀ s t, RealBodyBridge.norm s=RealBodyBridge.norm t → (stepStmts lc s).map RealBodyBridge.norm=(stepStmts rc t).map RealBodyBridge.norm) :
    (stepStmts (lp++lc) u).map RealBodyBridge.norm=(stepStmts (rp++rc) u).map RealBodyBridge.norm := by
  rw [ProgramRewrite.append,ProgramRewrite.append]
  cases hl : stepStmts lp u with
  | none =>
    cases hr : stepStmts rp u <;> simp_all
  | some s =>
    cases hr : stepStmts rp u with
    | none => simp [hl,hr] at preludes
    | some t =>
      have equal : RealBodyBridge.norm s=RealBodyBridge.norm t := by simpa [hl,hr] using preludes
      simpa only [Option.bind_some] using loops s t equal

/-- Complete executions agree on failure and every memory cell. Only dead
registers are normalized; no successful execution is an assumption. -/
theorem flat_of_prelude (kind : Kind) (A : FlatAlloc) (h : Call) (p : Phase)
    (scale : ℝ) (u : BlockState)
    (preludes : (stepStmts (A.flattenStmts (RealPrefixBridge.legacyPrefix kind h p scale)) u).map RealBodyBridge.norm =
      (stepStmts (A.flattenStmts (prelude kind h p scale)) u).map RealBodyBridge.norm) :
    (exec (A.flattenKernel (LegacySurface.source kind h p scale).eraseDType.toAlgKernel) u).map RealBodyBridge.norm =
      (exec (A.flattenKernel (Kernels.source kind h p scale).eraseDType.toAlgKernel) u).map RealBodyBridge.norm := by
  simp only [exec,FlatAlloc.flattenKernel,legacy_source_body,source_body,FlatAlloc.flattenStmts_append]
  rw [(loop_flat A kind h p scale).1,(loop_flat A kind h p scale).2]
  exact sequence _ _ _ _ u preludes (loop_equivalent kind h p scale)

theorem forward_flat (A : FlatAlloc) (h : Call) (p : Phase) (scale : ℝ) (u : BlockState) :
    (exec (A.flattenKernel (LegacySurface.source .forward h p scale).eraseDType.toAlgKernel) u).map RealBodyBridge.norm =
      (exec (A.flattenKernel (Kernels.source .forward h p scale).eraseDType.toAlgKernel) u).map RealBodyBridge.norm :=
  flat_of_prelude .forward A h p scale u (RealPrefixBridge.forward A h p scale u)

theorem flat (kind : Kind) (A : FlatAlloc) (h : Call) (p : Phase)
    (scale : ℝ) (positive : 0<p.span) (u : BlockState) :
    (exec (A.flattenKernel (LegacySurface.source kind h p scale).eraseDType.toAlgKernel) u).map RealBodyBridge.norm =
      (exec (A.flattenKernel (Kernels.source kind h p scale).eraseDType.toAlgKernel) u).map RealBodyBridge.norm := by
  apply flat_of_prelude
  cases kind
  · exact RealPrefixBridge.forward A h p scale u
  · exact RealPrefixBridge.prepare A h p scale positive u
  · exact RealPrefixBridge.backward A h p scale positive u

#axiomsClean forward_flat
#axiomsClean flat

end
end VeriTile.Bench.Repairs.DecayCumsum.RealSourceBridge
