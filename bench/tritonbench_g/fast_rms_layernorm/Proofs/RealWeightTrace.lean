import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightTrace
open VeriTile.Triton RealWeightProgram RealWeightOps RealWeightStep RealWeightSafety
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem prefix_safe (h : Call) (K : Nat) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (prefixCode h K) u := by
  rw [(bindings h K).1]
  apply TraceProofs.all_safe
  intro st member s
  simp only [initOps,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl <;> simp [Stmt.TraceSafe,Op.SafeAt.eq_def,colOp]

theorem tail_safe (h : Call) (K pid : Nat) (bounds : RegionBounds) (u : BlockState)
    (ready : Ready h K pid u) (inside : Bounds h K pid bounds) :
    Stmt.TraceSafeList bounds (tailCode h K) u := by
  rw [(bindings h K).2.2]
  obtain ⟨acc,ha⟩:=ready.acc
  simp [tail,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,Op.PointerAddressesSafeOn,Op.MemorySafe,
    colOp,maskOp,refOp,stepStmt,stepStmts,evalOp,evalOp.eq_def,ready.col,ready.mask,ha,
    columns,masks,Option.bind,Tile.bop,Tile.cop,Tile.vec,NumericDType.add,NumericDType.mul]
  exact inside.output

theorem complete (h : Call) (K : Nat) (bounds : RegionBounds) (u : BlockState)
    (inside : Bounds h K (u.pids 0) bounds) : Kernel.TraceSafe bounds (source h K) u := by
  let init:=initial h K u
  have ready : Ready h K (u.pids 0) init := by
    refine ⟨?_,?_,⟨fun _ => 0,?_⟩⟩ <;> simp [init,initial]
  obtain ⟨v,he,inv⟩:=RealWeightLoop.loop_executes h K (u.pids 0) u 0 init (RealWeightLoop.initial_holds h K u)
  have step : stepStmt (.forRangeDyn "row" (.constNat 0) (.constNat h.rows) (.constNat 1) (loopBody h K)) init=some v := by
    simpa [stepStmt,evalOp_constNat] using he
  have safe : Stmt.TraceSafe bounds
      (.forRangeDyn "row" (.constNat 0) (.constNat h.rows) (.constNat 1) (loopBody h K)) init := by
    simpa [Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp_constNat] using loop_safe h K (u.pids 0) bounds init ready inside
  change Stmt.TraceSafeList bounds (source h K).body u
  rw [source_body,List.append_assoc]
  apply TraceProofs.append_safe (prefixCode h K) u (prefix_safe h K bounds u)
  intro s hs
  have same : init=s := Option.some.inj ((prefix_executes h K u).symm.trans hs)
  subst s
  change Stmt.TraceSafeList bounds (.forRangeDyn _ _ _ _ _ :: tailCode h K) init
  rw [TraceProofs.cons_of_run_iff step]
  exact ⟨safe,tail_safe h K (u.pids 0) bounds v ⟨inv.col,inv.mask,⟨_,inv.acc⟩⟩ inside⟩

theorem flatten_ok (h : Call) (K : Nat) : (source h K).FlattenOk := by
  change StmtList.FlattenOk (source h K).body
  rw [source_body,(bindings h K).1,(bindings h K).2.1,(bindings h K).2.2]
  simp [StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def,initOps,body,tail,colOp,refOp,maskOp,loadOp,offsetOp]

#axiomsClean complete
#axiomsClean flatten_ok
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightTrace
