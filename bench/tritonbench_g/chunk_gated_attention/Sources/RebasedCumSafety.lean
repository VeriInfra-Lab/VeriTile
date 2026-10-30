import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumExecution
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedAccessSafety

/-! Finite allocation safety for the actual guarded cumulative source. Dummy
programs have no memory accesses; active programs use complete local tiles. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram
open RebasedCumAccess RebasedCumExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (c : Nat) (s o : RegionName) (bounds : RegionBounds) : Prop where
  input : RealHAccessSafety.MatrixBounds s (tileBase p l c) p.BT p.BS p.BT p.BS p.timeStride p.channelStride 0 0 bounds
  output : RealHAccessSafety.MatrixBounds o (tileBase p l c) p.BT p.BS p.BT p.BS p.timeStride p.channelStride 0 0 bounds

theorem compute_safe (p : Layout) (l : Launch) (c : Nat) (s o : RegionName) (swapped : Bool)
    (bounds : RegionBounds) (hb : Bounds p l c s o bounds) (u : BlockState)
    (his : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (RebasedCumProgram.computeCode p s o swapped) u := by
  simp only [RebasedCumProgram.computeCode,RealCumProgram.computeCode,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l s c swapped u his hih hit))]
  refine ⟨?_,?_⟩
  · cases swapped <;> simp [Stmt.TraceSafe,RebasedCumProgram.pointerOp,RebasedProgram.local2,baseOp,Op.SafeAt]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval p l o c swapped _
    (by simp [his]) (by simp [hih]) (by simp [hit])))]
  refine ⟨?_,?_⟩
  · cases swapped <;> simp [Stmt.TraceSafe,RebasedCumProgram.pointerOp,RebasedProgram.local2,baseOp,Op.SafeAt]
  apply TraceProofs.assign_cons (RealHAccessSafety.load2_safe s (tileBase p l c) p.BT p.BS p.BT p.BS
    p.timeStride p.channelStride 0 0 "p_s" bounds _ hb.input
    (by simp [pointer_eq,RebasedAccess.matrix]))
  intro bs
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro bo
  apply Stmt.TraceSafeList.cons_intro (RealHAccessSafety.store2_safe o (tileBase p l c) p.BT p.BS p.BT p.BS
    p.timeStride p.channelStride 0 0 "p_o" "b_o" bounds _ hb.output
    (by simp [pointer_eq,RebasedAccess.matrix]))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem active_safe (p : Layout) (s o : RegionName) (base : Nat) (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hb : Bounds p (decoded p base u) (chunk p base u) s o bounds) :
    Stmt.TraceSafeList bounds (RebasedCumProgram.activeCode p s o swapped) (linearState base u) := by
  rw [RebasedCumProgram.activeCode]
  apply TraceProofs.append_safe
  · apply TraceProofs.all_safe
    simp [RebasedCumProgram.decodeCode,initCode,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro v he
    rw [initializes] at he
    cases he
    exact compute_safe p (decoded p base u) (chunk p base u) s o swapped bounds hb (initial p base u)
      (by simp [initial]) (by simp [initial]) (by simp [initial])

theorem source_safe (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) (bounds : RegionBounds) (u : BlockState)
    (hb : enabled p heads base u → Bounds p (decoded p base u) (chunk p base u) s o bounds) :
    (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe,RebasedCumProgram.source_body,RebasedCumProgram.body]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (show
    evalOp (.add .nat .nil (.programId 0) (.constNat base)) u =
      some (Tile.scalar (linear base u)) from by
        simp only [evalOp_add,evalOp_programId,evalOp_constNat]; rfl))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds [.ifThen _ _] (linearState base u)
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt],?_⟩
  rw [test_eval]
  by_cases ha : enabled p heads base u
  · simp only [Tile.scalar,decide_eq_true ha,↓reduceIte]
    exact active_safe p s o base swapped bounds u (hb ha)
  · simp only [Tile.scalar,decide_eq_false ha,Bool.false_eq_true,↓reduceIte]

theorem source_flatten_ok (p : Layout) (s o : RegionName) (heads base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) :
    (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,RebasedCumProgram.source_body]
  cases swapped <;> simp [RebasedCumProgram.body,RebasedCumProgram.activeCode,RebasedCumProgram.decodeCode,
    initCode,RebasedCumProgram.computeCode,RealCumProgram.computeCode,RebasedCumProgram.pointerOp,
    RebasedProgram.local2,baseOp,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumSafety
