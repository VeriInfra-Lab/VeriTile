import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardState

/-! The first six actual loop statements, evaluated from the reached invariant.
The three input snapshots are internal frame facts, not prepared carry values. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardValues
open VeriTile.Triton Primitives MaskedMath ForwardState
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem key_load (p : Layout) (l : Launch) (initial u : BlockState)
    (r : RegionName) (ptr : RegName) (t : Nat)
    (hp : u.regs .ptr [p.BK] ptr = some (keyPointers p l r t))
    (hm : u.regs .bool [p.BK] "mask_bk" =
      some (maskTile p.BK (fun j => activeKey p l j.val)))
    (hmem : ∀ off, u.readMem r off = initial.readMem r off) :
    evalOp (ForwardSource.vectorLoad p.BK ptr "mask_bk") u =
      some (vector p.BK (keyValue p l initial r t)) := by
  simpa only [ForwardSource.vectorLoad, loadOp, keyPointers, keyValue, hmem] using
    (masked_load p.BK r ptr "mask_bk" (fun j => keyAddress p l t j.val)
      (fun j => activeKey p l j.val) u hp hm)

theorem value_load (p : Layout) (l : Launch) (initial u : BlockState)
    (r : RegionName) (ptr : RegName) (t : Nat)
    (hp : u.regs .ptr [p.BV] ptr = some (valuePointers p l r t))
    (hm : u.regs .bool [p.BV] "mask_bv" =
      some (maskTile p.BV (fun j => activeValue p l j.val)))
    (hmem : ∀ off, u.readMem r off = initial.readMem r off) :
    evalOp (ForwardSource.vectorLoad p.BV ptr "mask_bv") u =
      some (vector p.BV (valueValue p l initial r t)) := by
  simpa only [ForwardSource.vectorLoad, loadOp, valuePointers, valueValue, hmem] using
    (masked_load p.BV r ptr "mask_bv" (fun j => valueAddress p l t j.val)
      (fun j => activeValue p l j.val) u hp hm)

theorem before_store (p : Layout) (l : Launch) (initial : BlockState)
    (r : ForwardRegions) (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r hasSeed t u)
    (hqmem : ∀ off, u.readMem r.q off = initial.readMem r.q off)
    (hkmem : ∀ off, u.readMem r.k off = initial.readMem r.k off)
    (hvmem : ∀ off, u.readMem r.v off = initial.readMem r.v off) :
    stepStmts ((ForwardSource.loopBody p).take 6) u =
      some (beforeStore p l initial r hasSeed t u) := by
  let u1 := u.setReg "_k" .real [p.BK] (vector p.BK (keyValue p l initial r.k t))
  let u2 := u1.setReg "_v" .real [p.BV] (vector p.BV (valueValue p l initial r.v t))
  let u3 := u2.setReg "_q" .real [p.BK]
    (vector p.BK (fun j => keyValue p l initial r.q t j * p.scale))
  let u4 := u3.setReg "h" .real [p.BV,p.BK] (carryTile p l initial r hasSeed (t+1))
  let product := fun (j : Fin p.BV) (k : Fin p.BK) =>
    stateValue p l initial r.k r.v r.initial hasSeed (t+1) k j *
      (keyValue p l initial r.q t k * p.scale)
  let u5 := u4.setReg "_o" .real [p.BV,p.BK] (matrix p.BV p.BK product)
  have hk := key_load p l initial u r.k "p_k" t hu.k hu.maskK hkmem
  have hv := value_load p l initial u1 r.v "p_v" t
    (by simp [u1, hu.v]) (by simp [u1, hu.maskV]) (by simpa [u1] using hvmem)
  have hq := scale_eval p.BK (ForwardSource.vectorLoad p.BK "p_q" "mask_bk") u2
    (keyValue p l initial r.q t) p.scale
    (key_load p l initial u2 r.q "p_q" t
      (by simp [u2,u1,hu.q]) (by simp [u2,u1,hu.maskK])
      (by simpa [u2,u1] using hqmem))
  have hh : evalOp (ForwardSource.stateUpdate p) u3 =
      some (carryTile p l initial r hasSeed (t+1)) := by
    simpa only [ForwardSource.stateUpdate, stateOp, carryTile, state_succ] using
      (state_update p.BV p.BK u3 (decay p l)
        (fun j k => stateValue p l initial r.k r.v r.initial hasSeed t k j)
        (keyValue p l initial r.k t) (valueValue p l initial r.v t)
        (by simp [u3,u2,u1,hu.decay])
        (by simpa [u3,u2,u1,carryTile] using hu.carry)
        (by simp [u3,u2,u1]) (by simp [u3,u2]))
  have hp : evalOp (ForwardSource.outputProduct p) u4 =
      some (matrix p.BV p.BK product) := by
    exact output_product p.BV p.BK u4
      (fun j k => stateValue p l initial r.k r.v r.initial hasSeed (t+1) k j)
      (fun k => keyValue p l initial r.q t k * p.scale)
      (by simp [u4,carryTile]) (by simp [u4,u3])
  have hs := sum_rows p.BV p.BK u5 product (by simp [u5])
  dsimp only [ForwardSource.loopBody, List.take]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hk)]
  change stepStmts _ u1 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)]
  change stepStmts _ u2 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hq)]
  change stepStmts _ u3 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hh)]
  change stepStmts _ u4 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hp)]
  change stepStmts _ u5 = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape := [p.BV]) (name := "_o") hs)]
  rw [stepStmts.nil]
  have hout : (fun j : Fin p.BV => ∑ k : Fin p.BK, product j k) =
      MaskedMath.output p l initial r hasSeed t := by
    funext j
    rfl
  rw [hout]
  rfl

#axiomsClean key_load
#axiomsClean value_load
#axiomsClean before_store
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardValues
