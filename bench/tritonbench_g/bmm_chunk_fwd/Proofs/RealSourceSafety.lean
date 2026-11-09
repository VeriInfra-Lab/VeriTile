import bench.tritonbench_g.bmm_chunk_fwd.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open ControlArithmetic
attribute [local simp 2000] VeriTile.Triton.eval_truncDiv VeriTile.Triton.eval_truncMod
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
  simp (config := {maxSteps := 300000}) [CleanupReal.code,CleanupReal.kernel,Kernels.clearUpperKernel,clearUpperKernel,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType,Except.bind,eraseDType_float,Op.MemorySafe,Op.PointerAddressesSafeOn,Stmt.TraceSafeList,Stmt.TraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,
    MaskOpt.Active,stepStmt,evalOp.eq_def,bind,Option.bind,
    BlockState.withGridIndex,BlockState.withPids,BlockState.withNumPids,Host.Call.grid,Cleanup.config,
    Grid.rank,GridIndex.toPids,BlockState.setReg,Tile.ptrAdd,Tile.bop,Tile.cop,
    Tile.scalar,Tile.expandDim,TileShape.dropInsertedIndex,Region.cast,
    Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add,NumericDType.mul,NumericDType.sub,
    ComparableDType.lt,ComparableDType.le,BlockState.writeMemTyped_real,-Tile.scalar_eta]
  let h := parameters p
  have ceiling : h.shape.chunk+64-1=h.shape.chunk+63 := by omega
  simp only [CleanupSource.addresses,Cleanup.active,Cleanup.row,Cleanup.column,
    Cleanup.offset,Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,
    Nat.mul_add,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,and_assoc,ceiling]
  have hcdiv : Int.tdiv ((h.shape.chunk : Int)+63) 64 =
      (Host.cdiv h.shape.chunk 64 : Int) := by
    simpa only [Host.cdiv,ceiling,show (64 : Int)-1=63 by decide] using
      (ControlArithmetic.cdiv_signed_int h.shape.chunk 64)
  have row_cast (i : Fin 32) :
      (Cleanup.pid0 pid : Int)/(Host.cdiv h.shape.chunk 64 : Int)*32+(i.val : Int) =
        (Cleanup.row h pid i : Int) := by
    simp [Cleanup.row,Cleanup.pid0]
  have col_cast (j : Fin 64) :
      (Cleanup.pid0 pid : Int).tmod (Host.cdiv h.shape.chunk 64 : Int)*64+(j.val : Int) =
        (Cleanup.column h pid j : Int) := by
    simp [Cleanup.column,Cleanup.pid0,Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg _)]
  have offset_cast (i : Fin 32) (j : Fin 64) :
      Int.toNat (((Cleanup.pid1 pid : Int)*(h.shape.chunks*h.shape.groups : Int)+(Cleanup.pid2 pid : Int))*
        (h.shape.chunk : Int)*(h.shape.chunk : Int)+(Cleanup.row h pid i : Int)*(h.shape.chunk : Int)+
        (Cleanup.column h pid j : Int))=Cleanup.offset h pid i j := by
    have eq : ((Cleanup.pid1 pid : Int)*(h.shape.chunks*h.shape.groups : Int)+(Cleanup.pid2 pid : Int))*
        (h.shape.chunk : Int)*(h.shape.chunk : Int)+(Cleanup.row h pid i : Int)*(h.shape.chunk : Int)+
        (Cleanup.column h pid j : Int)=(Cleanup.offset h pid i j : Int) := by
      simp [Cleanup.offset,Cleanup.pid1,Cleanup.pid2,Nat.add_mul]
      ring
    rw [eq];rfl
  intro i j hr hc upper
  change (Cleanup.pid0 pid : Int)/(((h.shape.chunk : Int)+63).tdiv 64)*32+(i.val : Int)<(h.shape.chunk : Int) at hr
  change (Cleanup.pid0 pid : Int).tmod (((h.shape.chunk : Int)+63).tdiv 64)*64+(j.val : Int)<(h.shape.chunk : Int) at hc
  change (Cleanup.pid0 pid : Int)/(((h.shape.chunk : Int)+63).tdiv 64)*32+(i.val : Int)<
    (Cleanup.pid0 pid : Int).tmod (((h.shape.chunk : Int)+63).tdiv 64)*64+(j.val : Int) at upper
  have active : Cleanup.active h pid i j := by
    try simp only [hcdiv,row_cast,col_cast,Int.ofNat_lt] at hr hc upper
    exact ⟨hr,hc,upper⟩
  have bounded := output _ (Cleanup.offset_bound h pid i j active)
  have address : Int.toNat (((Cleanup.pid1 pid : Int)*(h.shape.chunks*h.shape.groups : Int)+
      (Cleanup.pid2 pid : Int))*(h.shape.chunk : Int)*(h.shape.chunk : Int)+
      ((Cleanup.pid0 pid : Int)/((h.shape.chunk : Int)+63).tdiv 64*32+(i.val : Int))*(h.shape.chunk : Int)+
      ((Cleanup.pid0 pid : Int).tmod (((h.shape.chunk : Int)+63).tdiv 64)*64+(j.val : Int)))=
        Cleanup.offset h pid i j := by
    rw [hcdiv,row_cast i,col_cast j]
    exact offset_cast i j
  change Int.toNat (((Cleanup.pid1 pid : Int)*(h.shape.chunks*h.shape.groups : Int)+
      (Cleanup.pid2 pid : Int))*(h.shape.chunk : Int)*(h.shape.chunk : Int)+
      ((Cleanup.pid0 pid : Int)/((h.shape.chunk : Int)+63).tdiv 64*32+(i.val : Int))*(h.shape.chunk : Int)+
      ((Cleanup.pid0 pid : Int).tmod (((h.shape.chunk : Int)+63).tdiv 64)*64+(j.val : Int)))<bounds h.Out
  rw [address]
  exact bounded

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
