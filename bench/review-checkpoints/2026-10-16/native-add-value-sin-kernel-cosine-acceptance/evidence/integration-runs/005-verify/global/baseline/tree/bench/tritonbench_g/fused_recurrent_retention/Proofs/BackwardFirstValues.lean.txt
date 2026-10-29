import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstState
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardValues

/-! The actual first backward loop loads, carry update and d_q reduction.
Its formula uses only the original memory snapshot and independent finite sums. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstValues
open VeriTile.Triton Primitives MaskedMath BackwardFirstState
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1100000
set_option linter.unusedSimpArgs false

theorem before_store (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r hasSeed t u)
    (hkmem : ∀ off, u.readMem r.k off = initial.readMem r.k off)
    (hvmem : ∀ off, u.readMem r.v off = initial.readMem r.v off)
    (hdmem : ∀ off, u.readMem r.do_ off = initial.readMem r.do_ off) :
    stepStmts ((BackwardSource.firstBody p).take 6) u =
      some (beforeStore p l initial r hasSeed t u) := by
  let u1 := u.setReg "_k" .real [p.BK] (vector p.BK (keyValue p l initial r.k t))
  let u2 := u1.setReg "_v" .real [p.BV] (vector p.BV (valueValue p l initial r.v t))
  let u3 := u2.setReg "_do" .real [p.BV] (vector p.BV (valueValue p l initial r.do_ t))
  let u4 := u3.setReg "h" .real [p.BK,p.BV] (carryTile p l initial r hasSeed (t+1))
  let product := fun (k : Fin p.BK) (j : Fin p.BV) =>
    stateValue p l initial r.k r.v r.initial hasSeed (t+1) k j * valueValue p l initial r.do_ t j
  let u5 := u4.setReg "_d_q" .real [p.BK,p.BV] (matrix p.BK p.BV product)
  have hk := ForwardValues.key_load p l initial u r.k "p_k" t hu.k hu.maskK hkmem
  have hv := ForwardValues.value_load p l initial u1 r.v "p_v" t
    (by simp [u1,hu.v]) (by simp [u1,hu.maskV]) (by simpa [u1] using hvmem)
  have hd := ForwardValues.value_load p l initial u2 r.do_ "p_do" t
    (by simp [u2,u1,hu.dout]) (by simp [u2,u1,hu.maskV]) (by simpa [u2,u1] using hdmem)
  have hh : evalOp (BackwardSource.stateUpdate p) u3 =
      some (carryTile p l initial r hasSeed (t+1)) := by
    simpa only [carryTile,state_succ] using
      (BackwardPrimitives.state_update p u3 (decay p l)
        (fun k j => stateValue p l initial r.k r.v r.initial hasSeed t k j)
        (keyValue p l initial r.k t) (valueValue p l initial r.v t)
        (by simp [u3,u2,u1,hu.decay]) (by simpa [u3,u2,u1,carryTile] using hu.carry)
        (by simp [u3,u2,u1]) (by simp [u3,u2]))
  have hp : evalOp (BackwardSource.dqProduct p) u4 = some (matrix p.BK p.BV product) := by
    exact BackwardPrimitives.dq_product p u4
      (fun k j => stateValue p l initial r.k r.v r.initial hasSeed (t+1) k j)
      (valueValue p l initial r.do_ t) (by simp [u4,carryTile]) (by simp [u4,u3])
  have hs := BackwardPrimitives.reduce_rows p.BK p.BV (.ref .real [p.BK,p.BV] "_d_q")
    u5 product (by simp [u5])
  have hscaled := scale_eval p.BK
    (.reduceSum ⟨1,by simp⟩ Bool.false (.ref .real [p.BK,p.BV] "_d_q")) u5
    (fun k => ∑ j : Fin p.BV,product k j) p.scale hs
  dsimp only [BackwardSource.firstBody,List.take]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hk)]
  change stepStmts _ u1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)]
  change stepStmts _ u2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hd)]
  change stepStmts _ u3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hh)]
  change stepStmts _ u4 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hp)]
  change stepStmts _ u5 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BK]) (name := "d_q") hscaled)]
  rw [stepStmts.nil]
  have hout : (fun k : Fin p.BK => (∑ j : Fin p.BV,product k j)*p.scale) =
      MaskedMath.dq p l initial r hasSeed t := by funext k; rfl
  rw [hout]
  rfl

#axiomsClean before_store
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstValues
