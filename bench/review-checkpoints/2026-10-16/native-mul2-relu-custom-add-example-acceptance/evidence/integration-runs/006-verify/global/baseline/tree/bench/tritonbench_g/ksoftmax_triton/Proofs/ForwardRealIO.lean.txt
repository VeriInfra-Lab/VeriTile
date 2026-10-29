import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealReadback

/-! Draft final assembly for root validation, after the source execution and
readback leaves. The declared input regions remain X and M; when the mask is
absent, the M read gate is false and its allocated extent may be zero.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem fp32_erased_projects (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) :
    (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgorithm? =
      .ok (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel := by
  simp only [fp32Source, ComputeKernel.eraseDType, Typed.fp32_projects]
  simp

theorem fp32_erased_flattenOk (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) :
    (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel.FlattenOk := by
  simp only [fp32Source, ComputeKernel.eraseDType, Typed.fp32_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases mode <;> cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [Typed.fp32Original, MaskMode.python, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
      ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      FloatDType.toTileDType, Region.cast]
    simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
    simp [Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Trace safety is proved from the source walk and actual read/write gates.
It is not inferred from equality of final states. -/
theorem fp32_erased_traceSafe (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * xm + s.pids 1 * xn + j.val < bounds X)
    (hm : ∀ j : Fin D, mode.present = Bool.true →
      readActive K (s.pids 1) causal j.val = Bool.true →
      maskAddress mode (s.pids 0) (s.pids 1) sm j.val < bounds M)
    (hy : ∀ j : Fin D, j.val < K →
      s.pids 0 * ym + s.pids 1 * yn + j.val < bounds Y) :
    Kernel.TraceSafe bounds
      (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType.toAlgKernel s := by
  simp only [fp32Source, ComputeKernel.eraseDType, Typed.fp32_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases mode <;> cases log <;> cases causal <;> cases fp16
  all_goals
    simp [readActive, MaskMode.present, maskAddress] at hx hm
    dsimp only [Typed.fp32Original, MaskMode.python, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
      ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      FloatDType.toTileDType, Region.cast]
    simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
    unfold Kernel.TraceSafe
    simp [Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      Op.PointerAddressesSafeOn, Op.MemorySafe, BlockState.setReg, tile_elementwise,
      Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, hD]
    first
    | exact ⟨hx, hy⟩
    | exact ⟨hx, hm, hy⟩

def fp32ErasedIO (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) : Masked2DKernelIO₂ where
  kernel := (fp32Source Y X M ym yn xm xn sm K D log mode causal fp16).eraseDType
  projection := fp32_erased_projects Y X M ym yn xm xn sm K D log mode causal fp16
  in1 := X
  in2 := M
  out := Y
  B := D
  read1 := fun batch row j => batch * xm + row * xn + j.val
  read2 := fun batch row j => maskAddress mode batch row sm j.val
  write := fun batch row j => batch * ym + row * yn + j.val
  mask := fun _ row j => readActive K row causal j.val = Bool.true
  read2Mask := fun _ row j =>
    mode.present = Bool.true ∧ readActive K row causal j.val = Bool.true
  writeMask := fun _ _ j => j.val < K

/-- Mathematical finite flat-memory IO for the complete original fp32
source. Native numerical admission and the candidate erasure bridge remain
separate obligations; this theorem describes this original source only. -/
specification ksoftmax_forward_fp32_real_correctness (Y X M : RegionName)
    (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) :
    fp32ErasedIO Y X M ym yn xm xn sm K D log mode causal fp16 ⊨
      fun _ row xs masks i => formula K row log causal mode xs masks i := by
  refine Masked2DKernelIO₂.Implements.intro _
    (fp32_erased_flattenOk Y X M ym yn xm xn sm K D log mode causal fp16) ?_ ?_
  · intro bounds s hx hm hy _
    exact fp32_erased_traceSafe Y X M ym yn xm xn sm K D hD log mode causal fp16
      bounds s hx (fun j hp hj => hm j ⟨hp, hj⟩) hy
  · intro s xs masks hx hm
    obtain ⟨t, hrun, hvalue, hframe⟩ :=
      fp32_source_region_run Y X M ym yn xm xn sm K D hD log mode causal fp16
        s xs masks hx (fun j hp hj => hm j ⟨hp, hj⟩)
    exact ⟨t, hrun, hvalue, fun r offset hmiss _ => hframe r offset hmiss⟩

#axiomsClean fp32_erased_projects
#axiomsClean fp32_erased_flattenOk
#axiomsClean fp32_erased_traceSafe
#axiomsClean ksoftmax_forward_fp32_real_correctness
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
