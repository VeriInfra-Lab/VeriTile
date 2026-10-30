import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardMath
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl

/-! Draft for root validation: generate an execution witness from the actual
complete fp32 source, with explicit mathematical dtype erasure. The mask enum
is a success precondition; unknown strings remain in the structural theorem.
No theorem in this file identifies real arithmetic with native IEEE results.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Every source flag and the original Option String tests remain in the
source. Only the interface restricts the successful-run mask domain. -/
def fp32Source (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) : ComputeKernel :=
  Typed.fp32Original Y X M ym yn xm xn sm K D log mode.python causal fp16

/-- Reduce the complete original source interpreter to produce its actual
final state. No postulated output tile or whole-run equation is supplied.
D > 0 is needed by the maximum even when every read lane is inactive. -/
noncomputable def fp32_source_runState (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (s : BlockState) :
    {t : BlockState //
      exec (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel
        s = some t} := by
  cases mode <;> cases log <;> cases causal <;> cases fp16
  all_goals
    refine ⟨?_, ?_⟩
    swap
    dsimp only [fp32Source]
    simp only [ComputeKernel.eraseDType, Typed.fp32_projects,
      ComputeKernel.toAlgKernel_fromKernelBody]
    dsimp only [Typed.fp32Original, MaskMode.python,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Except.bind, Except.pure, Bind.bind, Pure.pure,
      Kernel.eraseDType, ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      FloatDType.toTileDType, Region.cast]
    simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
    simp (maxSteps := 100000) [exec, stepStmts, stepStmt, evalOp.eq_def,
      tile_elementwise, Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, hD]
    try rfl

theorem fp32_source_exec_isSome (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) (s : BlockState) :
    ∃ t, exec (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel
      s = some t :=
  ⟨(fp32_source_runState Y X M ym yn xm xn sm K D hD log mode causal fp16 s).val,
    (fp32_source_runState Y X M ym yn xm xn sm K D hD log mode causal fp16 s).property⟩

#axiomsClean fp32_source_runState
#axiomsClean fp32_source_exec_isSome
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
