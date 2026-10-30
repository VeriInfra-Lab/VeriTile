import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealValues

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealStep
open VeriTile.Triton RealProgram IndexProgram RealValues
open VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep (outputTile scatter nextCarry chunkSuffix)
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def indexed (h : Call) (c : Config) (q f : Nat) (u : BlockState) : BlockState :=
  let u1 := u.setReg "start_t" .int [] (Tile.scalar (q*c.BT:Int))
  let u2 := u1.setReg "start_s" .int [] (Tile.scalar (f*BS:Int))
  let u3 := u2.setReg "lane_t" .nat [c.BT] (lanes c.BT)
  let u4 := u3.setReg "lane_s" .nat [BS] (lanes BS)
  let u5 := u4.setReg "active_t" .bool [c.BT] (maskT h c q)
  let u6 := u5.setReg "active_s" .bool [BS] (maskS h f)
  let u7 := u6.setReg "times" .int [c.BT] (times h c q)
  let u8 := u7.setReg "features" .int [BS] (features h f)
  u8.setReg "active" .bool [c.BT,BS] (mask h c q f)

def beforeStore (h : Call) (c : Config) (row q f : Nat) (a : Nat → ℝ)
    (acc : Fin BS → ℝ) (u : BlockState) : BlockState :=
  let u1 := (indexed h c q f u).setReg "p_s" .ptr [c.BT,BS] (pointer h c row q f h.input)
  let u2 := u1.setReg "p_z" .ptr [c.BT,BS] (pointer h c row q f h.output)
  let u3 := u2.setReg "b_s" .real [c.BT,BS] (inputTile h c row f a q)
  u3.setReg "b_c" .real [c.BT,BS] (outputTile (layout h c) (launch row f) a (fun _ => 0) q acc)

def result (h : Call) (c : Config) (row q f : Nat) (a : Nat → ℝ)
    (acc : Fin BS → ℝ) (u : BlockState) : BlockState :=
  (scatter (layout h c) (launch row f) h.output a (fun _ => 0) q acc
    (beforeStore h c row q f a acc u)).setReg "b_z" .real [BS]
    (Tile.vec fun j => some (nextCarry (layout h c) (launch row f) a (fun _ => 0) q acc j))

theorem output_eval (h : Call) (c : Config) (row q f : Nat) (a : Nat → ℝ) (acc : Fin BS → ℝ) (u : BlockState)
    (ha : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j)))
    (hm : u.regs .real [c.BT,c.BT] "m_s"=some (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.upperTriTile c.BT))
    (hb : u.regs .real [c.BT,BS] "b_s"=some (inputTile h c row f a q)) :
    evalOp (outputOp c) u=some (outputTile (layout h c) (launch row f) a (fun _ => 0) q acc) :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.add_dot_eval (layout h c) (launch row f) a (fun _ => 0) q acc u ha hm hb

theorem carry_eval (h : Call) (c : Config) (row q f : Nat) (a : Nat → ℝ) (acc : Fin BS → ℝ) (u : BlockState)
    (ha : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j)))
    (hb : u.regs .real [c.BT,BS] "b_s"=some (inputTile h c row f a q)) :
    evalOp (carryOp c) u=some (Tile.vec fun j => some (nextCarry (layout h c) (launch row f) a (fun _ => 0) q acc j)) :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.carry_eval (layout h c) (launch row f) a (fun _ => 0) q acc u ha hb

theorem store_eval (h : Call) (c : Config) (row q : Nat) (f : Geometry.FeatureBlock h)
    (a : Nat → ℝ) (acc : Fin BS → ℝ) (u : BlockState) (start : q*c.BT<h.T)
    (hp : u.regs .ptr [c.BT,BS] "p_z"=some (pointer h c row q f.val h.output))
    (hm : u.regs .bool [c.BT,BS] "active"=some (mask h c q f.val))
    (hv : u.regs .real [c.BT,BS] "b_c"=some (outputTile (layout h c) (launch row f.val) a (fun _ => 0) q acc)) :
    stepStmt (.store .real [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
      (.ref .real [c.BT,BS] "b_c") (.mask (.ref .bool [c.BT,BS] "active"))) u=
      some (scatter (layout h c) (launch row f.val) h.output a (fun _ => 0) q acc u) := by
  simp only [stepStmt,evalOp_ref,hp,hm,hv,Option.bind_some,Option.bind_eq_bind,
    Bool.true_and,pointer,mask,decide_eq_true_eq,Option.map_some]
  apply congrArg some
  unfold scatter
  apply congrArg (fun fn => (TileShape.allIndices [c.BT,BS]).foldl fn u)
  funext st idx
  rcases idx with ⟨i,j,⟨⟩⟩
  have equiv := Geometry.active_iff h c q f i j start
  by_cases ha : Geometry.active h c q f.val i j
  · have hv := equiv.mp ha
    simp [ha,Geometry.time,Geometry.feature,ha.1,ha.2,outputTile,Tile.mat,layout,launch,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,Geometry.address,hv.1,hv.2]
  · have hv := equiv.not.mp ha
    simp [ha,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active,layout,launch,hv]

def chunkCode (h : Call) (c : Config) : List Stmt := (loopBody h c).tail

theorem executes (h : Call) (c : Config) (row q : Nat) (f : Geometry.FeatureBlock h)
    (a : Nat → ℝ) (acc : Fin BS → ℝ) (u : BlockState) (start : q*c.BT<h.T)
    (his : u.regs .int [] "i_s"=some (Tile.scalar (f.val:Int)))
    (hih : u.regs .int [] "i_bh"=some (Tile.scalar (row:Int)))
    (hit : u.regs .int [] "i_t"=some (Tile.scalar (q:Int)))
    (read : ∀ off, u.readMem h.input off=a off)
    (htri : u.regs .real [c.BT,c.BT] "m_s"=some (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.upperTriTile c.BT))
    (hacc : u.regs .real [BS] "b_z"=some (Tile.vec fun j => some (acc j))) :
    stepStmts (chunkCode h c) u=some (result h c row q f.val a acc u) := by
  simp only [chunkCode,loopBody,List.tail_cons]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (startT_eval c q u hit))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (startS_eval f.val _ (by simp [his])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange c.BT) _=some (lanes c.BT) from by rw [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange BS) _=some (lanes BS) from by rw [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (activeT_eval h c q _ start (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (activeS_eval h f.val _ (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (time_eval h c q _ (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (feature_eval h f.val _ (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mask_eval h c q f.val _ (by simp) (by simp)))]
  change stepStmts _ (indexed h c q f.val u)=_
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval h c row q f.val h.input _
    (by simp [indexed,hih]) (by simp [indexed]) (by simp [indexed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval h c row q f.val h.output _
    (by simp [indexed,hih]) (by simp [indexed]) (by simp [indexed])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_eval h c row q f a _ start (by simp)
    (by simp [indexed]) (by intro off; simpa [indexed] using read off)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (output_eval h c row q f.val a acc _
    (by simp [indexed,hacc]) (by simp [indexed,htri]) (by simp)))]
  rw [stepStmts.cons_some (store_eval h c row q f a acc _ start (by simp) (by simp [indexed]) (by simp; rfl))]
  change stepStmts [.ifThen _ _] (scatter (layout h c) (launch row f.val) h.output a (fun _ => 0) q acc
    (beforeStore h c row q f.val a acc u))=_
  have hc : evalOp nonnegativeOp (scatter (layout h c) (launch row f.val) h.output a (fun _ => 0) q acc
      (beforeStore h c row q f.val a acc u))=some (Tile.scalar Bool.true) := by
    simp only [nonnegativeOp,evalOp]
    have hr : (scatter (layout h c) (launch row f.val) h.output a (fun _ => 0) q acc
      (beforeStore h c row q f.val a acc u)).regs .int [] "i_t"=some (Tile.scalar (q:Int)) := by
        simp [scatter,beforeStore,indexed,hit]
    rw [hr]
    apply congrArg some
    apply Tile.ext
    intro i
    change decide ((0:Int)≤(q:Int))=Bool.true
    simp
  rw [stepStmts,stepStmt,hc]
  simp only [Option.bind_some,Option.bind_eq_bind,Tile.scalar_data,↓reduceIte]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (carry_eval h c row q f.val a acc _
    (by simp [scatter,beforeStore,indexed,hacc]) (by simp [scatter,beforeStore])))]
  simp only [stepStmts.nil]
  rfl

#print axioms executes
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealStep
