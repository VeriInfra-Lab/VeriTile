import bench.tritonbench_g.layer_norm_liger.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Independent precomputed-mean stage, not an assumption about the mean
computed by the complete forward kernel. -/
def meanStore (dtype : FloatDType) (MeanPre Mean : RegionName) (Mean_row_stride : Nat) : ComputeKernel := triton {
  tl.region MeanPre = DTYPE, Mean = DTYPE
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row * $(Mean_row_stride))
  tl.store(Mean + row * $(Mean_row_stride), mean)
}
/-- Independent precomputed reciprocal-standard-deviation stage. -/
def rstdStore (dtype : FloatDType) (RSTDPre RSTD : RegionName) (RSTD_row_stride : Nat) : ComputeKernel := triton {
  tl.region RSTDPre = DTYPE, RSTD = DTYPE
  row = tl.program_id(0)
  rstd = tl.load(RSTDPre + row * $(RSTD_row_stride))
  tl.store(RSTD + row * $(RSTD_row_stride), rstd)
}

theorem mean_store_run {α : Type} [Inhabited α] (M : Algebra α)
    (pre output : RegionName) (stride : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (meanStore .fp32 pre output stride) s = some t ∧
      t.mem = (s.write output (s.pids 0*stride)
        (.mk .fp32 ((s.mem pre (s.pids 0*stride)).read .fp32))).mem := by
  simp [FP.Structural.exec, meanStore, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, numeric, bop, Region.cast, FloatDType.toTileDType]
  rfl

theorem rstd_store_run {α : Type} [Inhabited α] (M : Algebra α)
    (pre output : RegionName) (stride : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (rstdStore .fp32 pre output stride) s = some t ∧
      t.mem = (s.write output (s.pids 0*stride)
        (.mk .fp32 ((s.mem pre (s.pids 0*stride)).read .fp32))).mem := by
  simp [FP.Structural.exec, rstdStore, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, store, Option.bind, numeric, bop, Region.cast, FloatDType.toTileDType]
  rfl

#axiomsClean mean_store_run
#axiomsClean rstd_store_run
end
end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
