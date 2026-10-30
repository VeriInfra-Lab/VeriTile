import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealValues
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealStep
open VeriTile.Triton IndexProgram RealProgram RealValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def scatter (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState) : BlockState :=
  (TileShape.allIndices [c.BT,BS]).foldl (fun st idx =>
    if Geometry.active h c feature q idx.1.val idx.2.1.val then
      st.writeMem h.output
        (h.outputStrides.address h.H row (Geometry.time h c q idx.1.val) (Geometry.column h feature idx.2.1.val))
        (acc idx.2.1+chunkPrefix c f q idx.1 idx.2.1)
    else st) u

def beforeStore (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState) : BlockState :=
  let u1 := u.setReg "start_t" .int [] (Tile.scalar (q*c.BT))
  let u2 := u1.setReg "lanes_t" .int [c.BT] (lanes c.BT)
  let u3 := u2.setReg "active_t" .bool [c.BT] (axisMask h.T c.BT (q*c.BT))
  let u4 := u3.setReg "times" .int [c.BT] (coordinate h.T c.BT (q*c.BT))
  let u5 := u4.setReg "active" .bool [c.BT,BS] (mask h c feature q)
  let u6 := u5.setReg "p_s" .ptr [c.BT,BS] (pointer h c feature row q h.input h.inputStrides)
  let u7 := u6.setReg "p_z" .ptr [c.BT,BS] (pointer h c feature row q h.output h.outputStrides)
  let u8 := u7.setReg "b_s" .real [c.BT,BS] (inputTile c f q)
  u8.setReg "b_c" .real [c.BT,BS] (outputTile c f q acc)
def result (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState) : BlockState :=
  (scatter h c feature row q f acc (beforeStore h c feature row q f acc u)).setReg
    "b_z" .real [BS] (Tile.vec fun j => some (nextCarry c f q acc j))

theorem store_eval (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState)
    (hp : u.regs .ptr [c.BT,BS] "p_z"=some (pointer h c feature row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c feature q))
    (hv : u.regs .real [c.BT,BS] "b_c"=some (outputTile c f q acc)) :
    stepStmt (.store .real [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
      (.ref .real [c.BT,BS] "b_c") (.mask (.ref .bool [c.BT,BS] "active"))) u=
      some (scatter h c feature row q f acc u) := by
  simp only [stepStmt,evalOp_ref,hp,hm,hv,Option.bind_some,Option.bind_eq_bind,
    pointer,mask,outputTile,Tile.mat,Option.map_some,Option.bind_some,decide_eq_true_eq,BlockState.writeMemTyped]
  rfl

theorem executes (h : Call) (c : Config) (feature row q : Nat)
    (f : Nat → Fin BS → ℝ) (acc : Fin BS → ℝ) (u : BlockState)
    (ht : u.regs .int [] "i_t"=some (Tile.scalar q))
    (hb : u.regs .int [] "i_b"=some (Tile.scalar (row/h.H)))
    (hh : u.regs .int [] "i_h"=some (Tile.scalar (row%h.H)))
    (hs : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (feature*BS)))
    (hj : u.regs .int [BS] "columns"=some (coordinate h.S BS (feature*BS)))
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (lowerTriTile c.BT))
    (ha : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j)))
    (read : ∀ (i : Fin c.BT) (j : Fin BS),
      (if Geometry.active h c feature q i.val j.val then
        u.readMem h.input (h.inputStrides.address h.H row (Geometry.time h c q i.val) (Geometry.column h feature j.val))
      else 0)=f (q*c.BT+i.val) j) :
    stepStmts (loopBody h c) u=some (result h c feature row q f acc u) := by
  unfold loopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (timeStart_eval c q u ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.arange c.BT)) _=some (lanes c.BT) from by simp [evalOp,lanes,Tile.vec]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (active_eval h.T c.BT (q*c.BT) "lanes_t" "start_t" _ (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (coordinate_eval h.T c.BT (q*c.BT) "start_t" "active_t" "lanes_t" _ (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mask_eval h c feature q _ (by simp) (by simp [hs])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval h c feature row q h.input h.inputStrides _ (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval h c feature row q h.output h.outputStrides _ (by simp [hb]) (by simp [hh]) (by simp) (by simp [hj])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_eval h c feature row q f _ (by simp) (by simp) (by simpa using read)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (output_eval c f q acc _ (by simp [ha]) (by simp [hm]) (by simp)))]
  rw [stepStmts.cons_some (store_eval h c feature row q f acc _ (by simp) (by simp) (by simp))]
  change stepStmts [.ifThen _ _] (scatter h c feature row q f acc (beforeStore h c feature row q f acc u))=_
  have hc : evalOp (.ge .int .nil (.ref .int [] "i_t") (.constInt 0))
      (scatter h c feature row q f acc (beforeStore h c feature row q f acc u))=some (Tile.scalar Bool.true) := by
    have hr : (scatter h c feature row q f acc (beforeStore h c feature row q f acc u)).regs .int [] "i_t"=some (Tile.scalar q) := by simp [scatter,beforeStore,ht]
    rw [evalOp]
    simp only [evalOp_ref,hr,evalOp.eq_def,Option.bind_some,Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro i
    change decide ((0:Int)≤(q:Int))=Bool.true
    simp
  rw [stepStmts,stepStmt,hc]
  simp only [Option.bind_some,Option.bind_eq_bind,Tile.scalar_data,↓reduceIte]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (carry_eval c f q acc _
    (by simp [scatter,beforeStore,ha]) (by simp [scatter,beforeStore])))]
  simp only [stepStmts.nil]
  rfl

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealStep
