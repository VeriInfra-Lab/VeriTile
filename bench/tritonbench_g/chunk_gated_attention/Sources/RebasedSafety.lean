import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedLoopSafety
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedStateSafety

/-! Complete repaired-source allocation safety. Only enabled memory roles
and actual local parent addresses are required. Every reached iteration is
proved safe using the already-proved original numerical invariant. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RebasedAccessSafety
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (r : Regions) (gateK hasSeed storeFinal : Bool)
    (bounds : RegionBounds) : Prop where
  initial : hasSeed = Bool.true → StateBounds p l r.seed bounds
  loop : RebasedLoopSafety.Bounds p l r gateK bounds
  final : storeFinal = Bool.true → StateBounds p l r.final bounds

theorem source_safe (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool)
    (u : BlockState)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p (RebasedInit.decoded p base u) c)
    (hkey : (RebasedInit.decoded p base u).key*p.BK+p.BK ≤ p.K)
    (hvalue : (RebasedInit.decoded p base u).value*p.BV+p.BV ≤ p.V)
    (hs : RealHStep.Separate r)
    (hinj : RealHLoop.HistoryNoCollision p (RebasedInit.decoded p base u))
    (bounds : RegionBounds) (hb : Bounds p (RebasedInit.decoded p base u) r gateK hasSeed storeFinal bounds) :
    (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l := RebasedInit.decoded p base u
  let a : Inputs := ⟨u.readMem r.key,u.readMem r.value,u.readMem r.gate,u.readMem r.seed⟩
  let start := RebasedInit.initial p r a.seed base hasSeed u
  have hi := RebasedInit.invariant p r a base gateK hasSeed u (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  have hei := RebasedInit.executes p r a.seed base hasSeed u hkey hvalue (fun _ => rfl)
  have hloop := RebasedLoopSafety.loop_safe u p l r a gateK hasSeed swapped hfull hs hinj bounds hb.loop start hi
  obtain ⟨mid,hel,hm⟩ := RebasedLoop.executes u p l r a gateK hasSeed swapped hfull hs hinj start hi
  rw [Kernel.TraceSafe,RebasedProgram.source_body,RebasedProgram.body,List.append_assoc]
  apply TraceProofs.append_safe _ u (RebasedStateSafety.init_safe p r base hasSeed bounds u hb.initial)
  intro v hv
  rw [hei] at hv
  cases hv
  change Stmt.TraceSafeList bounds (RebasedLoop.statement p r gateK swapped :: RebasedProgram.finalization p r storeFinal) start
  rw [TraceProofs.cons_of_run_iff hel]
  exact ⟨hloop,RebasedStateSafety.final_safe p l r storeFinal bounds mid hb.final hm.key hm.value hm.head⟩

theorem source_flatten_ok (p : Layout) (r : Regions) (dot gate history : NativeSource.Storage)
    (seed final : FloatDType) (base : Nat) (gateK hasSeed storeFinal swapped : Bool) :
    (RebasedProgram.source p r dot gate history seed final base gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,RebasedProgram.source_body]
  cases swapped <;> simp [StmtList.FlattenOk,Stmt.FlattenOk,RebasedProgram.body,
    RebasedProgram.initialization,RebasedProgram.finalization,RebasedProgram.iteration,
    RebasedProgram.prelude,RebasedProgram.gateKey,RebasedProgram.gateValue,
    RealHProgram.iterationPrelude,RealHProgram.gateKey,RealHProgram.gateValue,
    updateStmt,storeCarry,carry,RealHProgram.dot,
    RebasedProgram.keyPtr,RebasedProgram.valuePtr,RebasedProgram.historyPtr,RebasedProgram.statePtr,
    RebasedProgram.local2,RebasedProgram.local1,RebasedProgram.endpointPtr,
    stateBase,times,load1,load2,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedSafety
