import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondState
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardValues

/-! Actual reverse-loop loads, += update, both gradient reductions and final
*=b update. All values are derived from the original memory snapshot. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondValues
open VeriTile.Triton Primitives MaskedMath BackwardSecondState
noncomputable section
set_option maxHeartbeats 1300000
set_option linter.unusedSimpArgs false

theorem before_stores (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (ht : t < p.T) (u : BlockState) (hu : Ready p l initial r (t+1) u)
    (hqmem : ∀ off, u.readMem r.q off = initial.readMem r.q off)
    (hkmem : ∀ off, u.readMem r.k off = initial.readMem r.k off)
    (hvmem : ∀ off, u.readMem r.v off = initial.readMem r.v off)
    (hdmem : ∀ off, u.readMem r.do_ off = initial.readMem r.do_ off) :
    stepStmts ((BackwardSource.secondBody p).take 8) u = some (beforeStores p l initial r t u) := by
  let u1 := u.setReg "_do" .real [p.BV] (vector p.BV (valueValue p l initial r.do_ t))
  let u2 := u1.setReg "_q" .real [p.BK] (vector p.BK (fun k => keyValue p l initial r.q t k*p.scale))
  let u3 := u2.setReg "_k" .real [p.BK] (vector p.BK (keyValue p l initial r.k t))
  let u4 := u3.setReg "_v" .real [p.BV] (vector p.BV (valueValue p l initial r.v t))
  let u5 := u4.setReg "d_h" .real [p.BK,p.BV] (dStateTile p l initial r t)
  let u6 := u5.setReg "d_k" .real [p.BK] (vector p.BK (MaskedMath.dk p l initial r t))
  let u7 := u6.setReg "d_v" .real [p.BV] (vector p.BV (MaskedMath.dv p l initial r t))
  have hd := ForwardValues.value_load p l initial u r.do_ "p_do" t
    (by simpa using hu.dout (by omega)) hu.maskV hdmem
  have hq := scale_eval p.BK (ForwardSource.vectorLoad p.BK "p_q" "mask_bk") u1
    (keyValue p l initial r.q t) p.scale
    (ForwardValues.key_load p l initial u1 r.q "p_q" t
      (by simpa [u1] using hu.q (by omega)) (by simp [u1,hu.maskK]) (by simpa [u1] using hqmem))
  have hk := ForwardValues.key_load p l initial u2 r.k "p_k" t
    (by simpa [u2,u1] using hu.k (by omega)) (by simp [u2,u1,hu.maskK])
    (by simpa [u2,u1] using hkmem)
  have hv := ForwardValues.value_load p l initial u3 r.v "p_v" t
    (by simpa [u3,u2,u1] using hu.v (by omega)) (by simp [u3,u2,u1,hu.maskV])
    (by simpa [u3,u2,u1] using hvmem)
  have hh : evalOp (BackwardSource.dStateUpdate p) u4 = some (dStateTile p l initial r t) := by
    have he := BackwardPrimitives.dstate_update p u4 (BackwardRemainingMath.carry p l initial r (t+1))
      (fun k => keyValue p l initial r.q t k*p.scale) (valueValue p l initial r.do_ t)
      (by simpa [u4,u3,u2,u1,carryTile] using hu.carry)
      (by simp [u4,u3,u2]) (by simp [u4,u3,u2,u1])
    have hmath : (fun k j => BackwardRemainingMath.carry p l initial r (t+1) k j +
        (keyValue p l initial r.q t k*p.scale)*valueValue p l initial r.do_ t j) =
        dStateValue p l initial r.q r.do_ t := by
      funext k j
      exact BackwardRemainingMath.current p l initial r t ht k j
    rw [hmath] at he
    exact he
  have hdk : evalOp (BackwardSource.dkOp p) u5 = some (vector p.BK (MaskedMath.dk p l initial r t)) := by
    exact BackwardSecondPrimitives.dk_eval p u5 (dStateValue p l initial r.q r.do_ t)
      (valueValue p l initial r.v t) (by simp [u5,dStateTile]) (by simp [u5,u4])
  have hdv : evalOp (BackwardSource.dvOp p) u6 = some (vector p.BV (MaskedMath.dv p l initial r t)) := by
    exact BackwardSecondPrimitives.dv_eval p u6 (dStateValue p l initial r.q r.do_ t)
      (keyValue p l initial r.k t) (by simp [u6,u5,dStateTile]) (by simp [u6,u5,u4,u3])
  have hdecay : evalOp (.mul .real .scalarR (.ref .real [p.BK,p.BV] "d_h")
      (.ref .real [] "b_b")) u7 = some (carryTile p l initial r t) := by
    have he := BackwardSecondPrimitives.decay_eval p u7 (dStateValue p l initial r.q r.do_ t)
      (decay p l) (by simp [u7,u6,u5,dStateTile]) (by simp [u7,u6,u5,u4,u3,u2,u1,hu.decay])
    have hmath : (fun k j => dStateValue p l initial r.q r.do_ t k j*decay p l) =
        BackwardRemainingMath.carry p l initial r t := by
      funext k j
      exact BackwardRemainingMath.next p l initial r t k j
    rw [hmath] at he
    exact he
  dsimp only [BackwardSource.secondBody,List.take]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hd)]
  change stepStmts _ u1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hq)]
  change stepStmts _ u2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hk)]
  change stepStmts _ u3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)]
  change stepStmts _ u4 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hh)]
  change stepStmts _ u5 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hdk)]
  change stepStmts _ u6 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hdv)]
  change stepStmts _ u7 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hdecay),stepStmts.nil]
  rfl

#axiomsClean before_stores
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondValues
