import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightLoop
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightMemory
open VeriTile.Triton RealWeightProgram RealWeightOps RealWeightStep RealWeightLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def scatter (h : Call) (K pid : Nat) (acc : Fin K → ℝ) (u : BlockState) : BlockState :=
  (TileShape.allIndices [K]).foldl (fun v i =>
    if pid*K+i.1.val<h.cols then v.writeMem "dw" (pid*K+i.1.val) (acc i.1) else v) u

theorem tail_executes (h : Call) (K pid : Nat) (u : BlockState) (acc : Fin K → ℝ)
    (hcol : u.regs .nat [K] "col"=some (columns K pid))
    (hmask : u.regs .bool [K] "mask"=some (masks h K pid))
    (hacc : u.regs .real [K] "acc"=some (Tile.vec fun j => some (acc j))) :
    stepStmts (tailCode h K) u=some (scatter h K pid acc u) := by
  rw [(bindings h K).2.2]
  simp [tail,colOp,refOp,maskOp,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,
    hcol,hmask,hacc,columns,masks,Tile.vec,scatter,BlockState.writeMemTyped]

def footprint (h : Call) (K pid : Nat) (q : RegionName) (o : Nat) : Prop :=
  q="dw" ∧ ∃ j : Fin K,pid*K+j.val<h.cols ∧ o=pid*K+j.val

theorem complete (h : Call) (K : Nat) (u : BlockState) :
    ∃ v,exec (source h K) u=some v ∧
      (∀ j : Fin K,u.pids 0*K+j.val<h.cols → v.readMem "dw" (u.pids 0*K+j.val)=
        carry h K (u.pids 0) u h.rows j) ∧
      (∀ q o,¬footprint h K (u.pids 0) q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,inv⟩ := loop_executes h K (u.pids 0) u 0 (RealWeightStep.initial h K u)
    (initial_holds h K u)
  let result := scatter h K (u.pids 0) (carry h K (u.pids 0) u h.rows) v
  refine ⟨result,?_,?_,?_⟩
  · change stepStmts (source h K).body u=some result
    rw [source_body,List.append_assoc,stepStmts.append_some (prefix_executes h K u)]
    change stepStmts (.forRangeDyn _ _ _ _ _ :: tailCode h K) _=_
    rw [stepStmts]
    simp only [stepStmt,evalOp_constNat,Option.bind_eq_bind,Option.bind_some,Tile.scalar_data,he]
    exact tail_executes h K (u.pids 0) v _ inv.col inv.mask inv.acc
  · intro j hj
    have injective : Function.Injective (fun i : TileIndex [K] => u.pids 0*K+i.1.val) := by
      intro a b eq
      exact Prod.ext (Fin.ext (Nat.add_left_cancel eq)) (Subsingleton.elim _ _)
    have readback := BlockState.scatter_readback_prop_masked_nd (region:="dw") v
      (fun i : TileIndex [K] => u.pids 0*K+i.1.val) (fun i => carry h K (u.pids 0) u h.rows i.1)
      (fun i => u.pids 0*K+i.1.val<h.cols) injective (j,PUnit.unit)
    simpa only [if_pos hj] using readback
  · intro q o outside
    have frame := BlockState.foldl_writeMem_prop_writeWithin (P:=fun a => footprint h K (u.pids 0) a.1 a.2)
      "dw" (fun i : TileIndex [K] => u.pids 0*K+i.1.val) (fun i => carry h K (u.pids 0) u h.rows i.1)
      (fun i => u.pids 0*K+i.1.val<h.cols) (TileShape.allIndices [K]) v
      (fun i _ hi => ⟨rfl,⟨i.1,hi,rfl⟩⟩)
    exact (frame q o outside).symm.trans (congrFun (congrFun inv.mem q) o)

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightMemory
