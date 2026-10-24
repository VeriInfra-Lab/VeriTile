import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopState
import VeriTile.Triton.KernelLemmas.LoopInvariant

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoopProgram
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def bound : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] "k_tiles") (.ref .nat [] "tiles_per_SM")
def loop (p : Parameters) : Stmt :=
  .forRangeDyn "_i" (.constNat 0) bound (.constNat 1) (RealProgram.body p)

/-- The full source is its actual sixteen-statement prefix followed by its
single flattened persistent range. No nested GEMM loop replaces that range. -/
theorem matches_source (p : Parameters) : (RealSource.kernel p Bool.false).body =
    RealPrefix.code p ++ [loop p] := by
  simp only [RealPrefix.code,loop,bound,RealProgram.body,RealSource.kernel,
    Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType]
  rfl

theorem bound_evaluates (p : Parameters) (s : BlockState) :
    evalOp bound (RealPrefix.result p s) = some (Tile.scalar (stop p (s.pids 0))) := by
  by_cases hd : s.pids 0 < RealMath.blocks p.M p.BM * RealMath.blocks p.N p.BN % p.SMS
  all_goals simp [bound,RealPrefix.result,hd,evalOp.eq_def,stop,work,count,
    RealPrefix.kTiles,RealPrefix.total,RealPrefix.rows,RealPrefix.columns,
    RealGroup.rows,RealGroup.columns,Schedule.tilesFor,Tile.scalar,Tile.bop,
    NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex,bind,Option.bind,
    -Tile.scalar_eta]

/-- A checked body invariant composes with the genuine runtime loop bound and
the actual prefix, starting from an arbitrary register file. -/
theorem executes_from_invariant (p : Parameters) (s : BlockState)
    (P : Nat → BlockState → Prop)
    (hinit : P 0 (RealPrefix.result p s))
    (hbound : ∀ i t, P i t → i≤stop p (s.pids 0))
    (hstep : ∀ i t, i<stop p (s.pids 0) → P i t →
      ∃ u, stepStmts (RealProgram.body p) (t.setReg "_i" .nat [] (Tile.scalar i)) = some u ∧ P (i+1) u) :
    ∃ t, stepStmts (RealSource.kernel p Bool.false).body s = some t ∧ P (stop p (s.pids 0)) t := by
  obtain ⟨n,t,he,hn,hP⟩ := forRangeDyn_inv
    (idx := "_i") (body := RealProgram.body p) (P := P)
    (show evalOp (.constNat 0) (RealPrefix.result p s) = some (Tile.scalar 0) by simp [evalOp.eq_def])
    (bound_evaluates p s)
    (show evalOp (.constNat 1) (RealPrefix.result p s) = some (Tile.scalar 1) by simp [evalOp.eq_def])
    (by decide) hinit hstep
  have hfinal : n=stop p (s.pids 0) := Nat.le_antisymm (hbound n t hP) hn
  subst n
  refine ⟨t,?_,hP⟩
  rw [matches_source,stepStmts.append_some (RealPrefix.executes p s)]
  change stepStmt (loop p) (RealPrefix.result p s) = some t at he
  simp only [stepStmts,he]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoopProgram
