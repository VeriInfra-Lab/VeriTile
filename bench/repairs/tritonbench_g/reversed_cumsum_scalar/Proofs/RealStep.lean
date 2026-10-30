import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.RealMemory
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealStep

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.RealStep
open VeriTile.Triton RealValues RealProgram
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev index_eval := VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealStep.index_eval

def beforeStore (h : Call) (c : Config) (row q : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) : BlockState :=
  let u0 := u.setReg "i_t" .int [] (Tile.scalar (q:Int))
  let u1 := u0.setReg "start_t" .int [] (Tile.scalar (q*c.BT:Int))
  let u2 := u1.setReg "lanes" .int [c.BT] (lanes c)
  let u3 := u2.setReg "active" .bool [c.BT] (mask h c q)
  let u4 := u3.setReg "times" .int [c.BT] (times h c q)
  let u5 := u4.setReg "p_s" .ptr [c.BT] (pointer h c row q h.input h.inputStrides)
  let u6 := u5.setReg "p_o" .ptr [c.BT] (pointer h c row q h.output h.outputStrides)
  let u7 := u6.setReg "b_s" .real [c.BT] (inputTile c.BT q f)
  let u8 := u7.setReg "b_zz" .real [] (Tile.scalar (some (chunkSum c.BT q f)))
  let u9 := u8.setReg "b_z" .real [] (Tile.scalar (some (acc+chunkSum c.BT q f)))
  u9.setReg "b_o" .real [c.BT] (outputTile c.BT q f (acc+chunkSum c.BT q f))

def result (h : Call) (c : Config) (row q : Nat)
    (f : Nat → ℝ) (acc : ℝ) (u : BlockState) : BlockState :=
  RealMemory.scatter h c row q f (acc+chunkSum c.BT q f) (beforeStore h c row q f acc u)

theorem executes (h : Call) (c : Config) (row : Nat) (input : Nat → ℝ)
    (acc : ℝ) (j : Int) (u : BlockState) (hj : j<1)
    (start : (-j).toNat*c.BT<h.T)
    (hr : u.regs .int [] "__rev_i_t"=some (Tile.scalar j))
    (hh : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int)))
    (ha : u.regs .real [] "b_z"=some (Tile.scalar (some acc)))
    (hm : ∀ t, t<h.T → u.readMem h.input (h.inputStrides.address h.H row t)=input t) :
    stepStmts (loopBody h c) u=some (result h c row (-j).toNat (value h input) acc u) := by
  unfold loopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (index_eval j u hj hr))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealValues.start_eval c (-j).toNat _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.arange c.BT)) _=some (lanes c) from by simp only [evalOp, Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (active_eval h c (-j).toNat _ start (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (time_eval h c (-j).toNat _ (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval h c row (-j).toNat h.input h.inputStrides _ (by simp [hh]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval h c row (-j).toNat h.output h.outputStrides _ (by simp [hh]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_eval h c row (-j).toNat input _ start (by simp) (by simp) (by intro t ht; simpa using hm t ht)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (shape:=[]) (sum_eval c.BT (-j).toNat (value h input) _ (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (carry_eval c.BT (-j).toNat (value h input) acc _ (by simp [ha]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (output_eval c.BT (-j).toNat (value h input) (acc+chunkSum c.BT (-j).toNat (value h input)) _ (by simp) (by simp)))]
  rw [stepStmts.cons_some (RealMemory.store_eval h c row (-j).toNat (value h input) (acc+chunkSum c.BT (-j).toNat (value h input)) _ (by simp) (by simp) (by simp))]
  rw [stepStmts.nil]
  rfl

#print axioms executes
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.RealStep
