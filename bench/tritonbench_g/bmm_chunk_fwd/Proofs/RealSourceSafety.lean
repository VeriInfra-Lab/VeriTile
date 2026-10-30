import bench.tritonbench_g.bmm_chunk_fwd.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem main_source_eq (p : Parameters) (c : Host.Config) :
    (mainSource p c).kernel=RealProgram.kernel Bool.false c.M c.N c.K (parameters p).parameters := by
  change (Kernels.fp32Source (parameters p) c).eraseDType=_
  rw [Kernels.fp32Source_eq (parameters p) c rfl rfl]
  exact RealProgram.storage_erasure .fp32 .fp32 Bool.false .fp32 c.M c.N c.K (parameters p).parameters

theorem main_safe (p : Parameters) (c : Host.Config) (v : RealInputs p) (s : BlockState)
    (nonempty : p.shape.outputSize≠0) (bounds : RegionBounds)
    (bounded : (realIO p c).InBounds bounds s v) :
    (AlgorithmStage.ordinary (mainSource p c) ((parameters p).grid c)).Safe bounds s := by
  intro pid
  rw [main_source_eq]
  exact RealSafety.safe Bool.false (s.withGridIndex pid) c.M c.N c.K (parameters p).parameters bounds
    (RealContract.bounds_of_views Bool.false _ _ _ _ _ bounds (input_bounds p c v s nonempty bounds bounded pid))

@[simp] theorem cleanup_initial (p : Parameters) (pid : Cleanup.PID (parameters p)) (s : BlockState) :
    (cleanupLaunch p pid).initial s=s.withGridIndex pid := rfl

/-- The actual cleanup source has no loads. Its single masked store uses the
already proved exact upper-triangle address image. -/
theorem cleanup_safe (p : Parameters) (pid : Cleanup.PID (parameters p))
    (bounds : RegionBounds) (output : ∀ off,off<p.shape.outputSize → off<bounds p.Out)
    (s : BlockState) :
    (AlgorithmStage.single (cleanupSource p) (cleanupLaunch p pid)).Safe bounds s := by
  change Kernel.TraceSafe bounds (CleanupReal.kernel (parameters p)) (s.withGridIndex pid)
  unfold Kernel.TraceSafe
  rw [CleanupReal.kernel_body]
  simp (config := {maxSteps := 300000}) [CleanupReal.code,Stmt.TraceSafeList,Stmt.TraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,
    MaskOpt.Active,stepStmt,evalOp.eq_def,bind,Option.bind,
    BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,Host.Call.grid,Cleanup.config,
    Grid.rank,GridIndex.toPids,BlockState.setReg,Tile.ptrAdd,Tile.bop,Tile.cop,
    Tile.scalar,Tile.expandDim,TileShape.dropInsertedIndex,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,
    ComparableDType.lt,ComparableDType.le,BlockState.writeMemTyped_real,-Tile.scalar_eta]
  intro i j hr hc upper
  have active : Cleanup.active (parameters p) pid i j := ⟨hr,hc,upper⟩
  have bounded := output _ (Cleanup.offset_bound (parameters p) pid i j active)
  simpa [Cleanup.offset,Cleanup.row,Cleanup.column,Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,
    NumericDType.div,NumericDType.sub,Nat.add_mul,Nat.mul_assoc] using bounded

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
