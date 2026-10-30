import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeSourceSafety
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeCandidateSafety
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeReferenceConnection
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeScheduleExecution

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_bounds (rest : Algebra Expr) (p : Parameters) (c : Host.Config)
    (v : Values p) (bounds : RegionName → Nat) (bounded : (io rest p c).toProgramIO.InBounds bounds v)
    (nonempty : p.shape.outputSize≠0) (pid : Host.PID (parameters p) c)
    (active : NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true) :
    SourceSafety.Bounds (parameters p) c (coordinates p c pid) bounds := by
  constructor
  · intro q hq i e mask
    have present : readMask p c 0 (SourcePointers.aOffset (parameters p) c (coordinates p c pid) q i e) :=
      ⟨nonempty,pid,active,⟨q,hq⟩,i,e,mask,rfl⟩
    exact bounded.2.1 (0 : Fin 4) (⟨_,read_extent p c 0 _ present⟩,PUnit.unit) present
  · intro q hq e j mask
    have present : readMask p c 1 (SourcePointers.bOffset (parameters p) c (coordinates p c pid) q e j) :=
      ⟨nonempty,pid,active,⟨q,hq⟩,e,j,mask,rfl⟩
    exact bounded.2.1 (1 : Fin 4) (⟨_,read_extent p c 1 _ present⟩,PUnit.unit) present
  · intro seq i mask
    have present : readMask p c 2 (SourcePointers.seqOffset (parameters p) (coordinates p c pid)
        (SourcePointers.row c (coordinates p c pid) i)) :=
      ⟨nonempty,seq,pid,active,Or.inl ⟨i,mask,rfl⟩⟩
    exact bounded.2.1 (2 : Fin 4) (⟨_,read_extent p c 2 _ present⟩,PUnit.unit) present
  · intro seq j mask
    have present : readMask p c 2 (SourcePointers.seqOffset (parameters p) (coordinates p c pid)
        (SourcePointers.col c (coordinates p c pid) j)) :=
      ⟨nonempty,seq,pid,active,Or.inr ⟨j,mask,rfl⟩⟩
    exact bounded.2.1 (2 : Fin 4) (⟨_,read_extent p c 2 _ present⟩,PUnit.unit) present
  · intro i j mask
    have present : writeMask p c (SourcePointers.outOffset (parameters p) c (coordinates p c pid) i j) :=
      ⟨nonempty,Or.inl ⟨pid,active,i,j,mask,rfl⟩⟩
    exact bounded.2.2.1 (0 : Fin 1)
      (⟨_,output_address_bound p c pid active i j mask⟩,PUnit.unit) present

theorem main_safe (rest : Algebra Expr) (p : Parameters) (c : Host.Config)
    (v : Values p) (bounds : RegionName → Nat) (bounded : (io rest p c).toProgramIO.InBounds bounds v)
    (nonempty : p.shape.outputSize≠0) (version : Bool) (pid : Host.PID (parameters p) c) (s : State Expr) :
    (mainInvocation rest p c version pid).Safe bounds s := by
  have original : TraceSafe (engine (backend rest c).algebra) bounds
      (Kernels.fp32Source (parameters p) c) (FPGrid.atIndex s pid) := by
    rw [Kernels.fp32Source_eq (parameters p) c rfl rfl]
    exact SourceSafety.program (backend rest c) (parameters p) c bounds (FPGrid.atIndex s pid)
      (source_bounds rest p c v bounds bounded nonempty pid) rfl rfl
  cases version
  · exact original
  · exact candidate_safe (backend rest c).algebra (parameters p) c bounds (FPGrid.atIndex s pid) original

/-- Actual cleanup source safety, using exactly its upper-triangle stores. -/
theorem cleanup_safe (rest : Algebra Expr) (p : Parameters) (pid : Cleanup.PID (parameters p))
    (bounds : RegionName → Nat)
    (output : ∀ off,Cleanup.Upper (parameters p) off → off<bounds p.Out) (s : State Expr) :
    (cleanupInvocation rest p pid).Safe bounds s := by
  change TraceSafe (engine (CleanupSource.algebra rest (.leaf .fp32 0))) bounds
    (Kernels.clearUpper p.Out p.shape.chunk (p.shape.chunks*p.shape.groups))
    (CleanupSource.atIndex (parameters p) s pid)
  simp (config := {maxSteps := 400000}) [TraceSafe,Kernels.clearUpper,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeAlg,Safety.safeOp.eq_def,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    State.setReg,integralFloorDiv,integralMod,CleanupSource.atIndex,CleanupSource.algebra,FP.Structural.store,
    GridIndex.toPids,Host.Call.grid,Cleanup.config,Grid.rank]
  intro i j hr hc upper
  have active : Cleanup.active (parameters p) pid i j := ⟨hr,hc,upper⟩
  have bounded := output _ ((Cleanup.image (parameters p) _).mp
    ((Cleanup.member (parameters p) _).mpr ⟨pid,i,j,active,rfl⟩))
  simpa [Cleanup.offset,Cleanup.row,Cleanup.column,Cleanup.pid0,Cleanup.pid1,Cleanup.pid2,Host.cdiv,
    parameters,Nat.add_mul,Nat.mul_assoc] using bounded

theorem all_safe (rest : Algebra Expr) (p : Parameters) (c : Host.Config)
    (v : Values p) (bounds : RegionName → Nat) (bounded : (io rest p c).toProgramIO.InBounds bounds v)
    (version : Bool) (s : State Expr) : ScheduleSafe bounds (schedule rest p c version) s := by
  by_cases empty : p.shape.outputSize=0
  · simp [schedule,empty,ScheduleSafe]
  · rw [schedule,if_neg empty,scheduleSafe_append]
    constructor
    · have all : ∀ (order : List (Host.PID (parameters p) c)) (s : State Expr),
          ScheduleSafe bounds (order.map (mainInvocation rest p c version)) s := by
        intro order
        induction order with
        | nil => intro s; trivial
        | cons pid order ih =>
          intro s
          exact ⟨main_safe rest p c v bounds bounded empty version pid s,fun next _=>ih next⟩
      exact all _ s
    · intro final runMain
      cases causal : p.causal
      · simp [ScheduleSafe]
      · have output : ∀ off,Cleanup.Upper (parameters p) off → off<bounds p.Out := by
          intro off upper
          obtain ⟨pid,i,j,active,eq⟩ := (Cleanup.member (parameters p) off).mp ((Cleanup.image (parameters p) off).mpr upper)
          have bound : off<p.shape.outputSize := by rw [eq]; exact Cleanup.offset_bound (parameters p) pid i j active
          exact bounded.2.2.1 (0 : Fin 1) (⟨off,bound⟩,PUnit.unit) ⟨empty,Or.inr ⟨causal,upper⟩⟩
        have all : ∀ (order : List (Cleanup.PID (parameters p))) (s : State Expr),
            ScheduleSafe bounds (order.map (cleanupInvocation rest p)) s := by
          intro order
          induction order with
          | nil => intro s; trivial
          | cons pid order ih =>
            intro s
            exact ⟨cleanup_safe rest p pid bounds output s,fun next _=>ih next⟩
        exact all _ final

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
