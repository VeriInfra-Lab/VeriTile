import bench.tritonbench_g.fused_recurrent_retention.Proofs.HeaderPrimitives

/-! Full source header execution from arbitrary initial registers. Both key
comparison spellings establish the same actual masks, addresses and zero state. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardHeader
open VeriTile.Triton Primitives MaskedMath ForwardState StatePrimitives ForwardHeaderSource
noncomputable section
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

def state (p : Layout) (r : ForwardRegions) (initial : BlockState) : BlockState :=
  let l := launch initial
  let u1 := initial.setReg "i_v" .nat [] (Tile.scalar l.value)
  let u2 := u1.setReg "i_k" .nat [] (Tile.scalar l.key)
  let u3 := u2.setReg "i_bh" .nat [] (Tile.scalar l.head)
  let u4 := u3.setReg "i_h" .nat [] (Tile.scalar (l.head % p.H))
  let u5 := u4.setReg "b_b" .real [] (Tile.scalar (some (MaskedMath.decay p l)))
  let u6 := u5.setReg "p_q" .ptr [p.BK] (keyPointers p l r.q 0)
  let u7 := u6.setReg "p_k" .ptr [p.BK] (keyPointers p l r.k 0)
  let u8 := u7.setReg "p_v" .ptr [p.BV] (valuePointers p l r.v 0)
  let u9 := u8.setReg "p_o" .ptr [p.BV] (outputPointers p l r.o 0)
  let u10 := u9.setReg "mask_bk" .bool [p.BK] (maskTile p.BK (fun j => activeKey p l j.val))
  let u11 := u10.setReg "mask_bv" .bool [p.BV] (maskTile p.BV (fun j => activeValue p l j.val))
  let u12 := u11.setReg "mask_kv" .bool [p.BV,p.BK] (stateMask p l)
  u12.setReg "h" .real [p.BV,p.BK] (matrix p.BV p.BK (fun _ _ => 0))

theorem executes (p : Layout) (r : ForwardRegions) (swapped : Bool) (initial : BlockState) :
    stepStmts (header p r swapped) initial = some (state p r initial) := by
  unfold header
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 initial))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.head_index p (launch initial) _ (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.decay p (launch initial) _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.pointer p.BK p.qHead (launch initial).head (launch initial).key "i_k" r.q _
      (by simp [launch]) (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.pointer p.BK p.qHead (launch initial).head (launch initial).key "i_k" r.k _
      (by simp [launch]) (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.pointer p.BV p.vHead (launch initial).head (launch initial).value "i_v" r.v _
      (by simp [launch]) (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.output_pointer p (launch initial) r.o _
      (by simp [launch]) (by simp [launch]) (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.key_mask p (launch initial) swapped _ (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.value_mask p (launch initial) _ (by simp [launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.state_mask p (launch initial) _ (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.full [p.BV,p.BK] (.const 0)) _ =
      some (matrix p.BV p.BK (fun _ _ => 0)) from by
        simp only [evalOp_full,evalOp_const,Option.bind,bind,Option.pure_def]
        rfl))]
  simp only [stepStmts.nil,state,launch,keyPointers,keyAddress,valuePointers,valueAddress,
    Nat.zero_mul,Nat.add_zero,BlockState.setReg_pids]

theorem ready (p : Layout) (r : ForwardRegions) (initial : BlockState) :
    Ready p (launch initial) initial r Bool.false 0 (state p r initial) := by
  constructor <;> simp [state,StatePrimitives.unseeded_zero,StatePrimitives.stateMask]

theorem memory (p : Layout) (r : ForwardRegions) (initial : BlockState) :
    (state p r initial).mem = initial.mem := rfl

#axiomsClean executes
#axiomsClean ready
#axiomsClean memory
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardHeader
