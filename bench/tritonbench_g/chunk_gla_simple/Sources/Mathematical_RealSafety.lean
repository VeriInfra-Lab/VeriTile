import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_RealExecution
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealOutputSafety

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Safe
open VeriTile.Triton Math Real
open ChunkGlaFwd.RealOutputSafety (MatrixBounds pointer load_safe store_safe)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (bounds : RegionBounds) : Prop where
  query : MatrixBounds r.query (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime 1 (l.chunk*p.BT) (kb*p.BK) bounds
  key : MatrixBounds r.key (l.head*p.keyHead) p.K p.T p.BK p.BT 1 p.keyTime (kb*p.BK) (l.chunk*p.BT) bounds
  history : MatrixBounds r.history (l.head*p.historyHead+l.chunk*p.K*p.V) p.K p.V p.BK p.BV p.historyKey 1
    (kb*p.BK) (l.value*p.BV) bounds

structure TailBounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  gate : ∀ i : Fin p.BT, row p l i.val < p.T → gateAddress p l i.val < bounds r.gate
  value : MatrixBounds r.value (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime 1 (l.chunk*p.BT) (l.value*p.BV) bounds
  output : MatrixBounds r.output (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime 1 (l.chunk*p.BT) (l.value*p.BV) bounds

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  loop : ∀ kb, kb < numBlocks p → StepBounds p l r kb bounds
  tail : TailBounds p l r bounds

theorem load_gate_safe (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) (u : BlockState)
    (hb : ∀ i : Fin p.BT, row p l i.val < p.T → gateAddress p l i.val < bounds r.gate)
    (hp : u.regs .blockPtr [p.BT] "p_g" = some
      ⟨fun _ => BlockPtr.mk r.gate (l.head*p.T) [p.T] [p.BT] [1] [l.chunk*p.BT]⟩) :
    (Op.load .real (.blockPtr (.ref .blockPtr [p.BT] "p_g") [0]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,⟨⟩⟩
  simpa only [TileShape.indexToList,BlockPtr.inBounds_1d,BlockPtr.address_1d,
    decide_eq_true_eq,Nat.mul_one,row,gateAddress] using hb i

theorem step_safe (p : Layout) (l : Launch) (r : Regions) (kb : Nat)
    (bounds : RegionBounds) (hb : StepBounds p l r kb bounds) (sl : BlockState)
    (hiv : sl.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hit : sl.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hibh : sl.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hik : sl.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    Stmt.TraceSafeList bounds (iteration p r) sl := by
  have hbaseH : ∀ st : BlockState, st.regs .nat [] "i_bh" = some (Tile.scalar l.head) →
      st.regs .nat [] "i_t" = some (Tile.scalar l.chunk) →
      evalOp (Op.add .nat .nil
        (Op.mul .nat .nil (Op.ref .nat [] "i_bh") (Op.constNat p.historyHead))
        (Op.mul .nat .nil (Op.mul .nat .nil (Op.ref .nat [] "i_t") (Op.constNat p.K)) (Op.constNat p.V))) st
      = some (Tile.scalar (l.head*p.historyHead+l.chunk*p.K*p.V)) := by
    intro st hbh ht
    rw [evalOp_add,mulConst_eval st "i_bh" _ _ hbh,evalOp_mul,mulConst_eval st "i_t" _ _ ht]
    simp only [evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
    rfl
  unfold iteration chunkGlaLoopBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.query sl _ _ _ [p.T,p.K] [p.BT,p.BK] [p.keyTime,1] (l.head*p.keyHead)
      (l.chunk*p.BT) (kb*p.BK) (mulConst_eval sl "i_bh" _ _ hibh)
      (mulConst_eval sl "i_t" _ _ hit) (mulConst_eval sl "i_k" _ _ hik)))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.key _ _ _ _ [p.K,p.T] [p.BK,p.BT] [1,p.keyTime] (l.head*p.keyHead)
      (kb*p.BK) (l.chunk*p.BT)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_t" _ _ (by simp [hit]))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.history _ _ _ _ [p.K,p.V] [p.BK,p.BV] [p.historyKey,1]
      (l.head*p.historyHead+l.chunk*p.K*p.V) (kb*p.BK) (l.value*p.BV)
      (hbaseH _ (by simp [hibh]) (by simp [hit])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load_safe r.query (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime 1
    (l.chunk*p.BT) (kb*p.BK) "p_q" bounds _ hb.query (by simp [pointer]))
  intro query
  apply TraceProofs.assign_cons (load_safe r.key (l.head*p.keyHead) p.K p.T p.BK p.BT 1 p.keyTime
    (kb*p.BK) (l.chunk*p.BT) "p_k" bounds _ hb.key (by simp [pointer]))
  intro key
  apply TraceProofs.assign_cons (load_safe r.history (l.head*p.historyHead+l.chunk*p.K*p.V) p.K p.V p.BK p.BV p.historyKey 1
    (kb*p.BK) (l.value*p.BV) "p_h" bounds _ hb.history (by simp [pointer]))
  intro history
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Safe

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Safe
open VeriTile.Triton Math Real
open ChunkGlaFwd.RealOutputSafety (MatrixBounds pointer load_safe store_safe)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Safe
