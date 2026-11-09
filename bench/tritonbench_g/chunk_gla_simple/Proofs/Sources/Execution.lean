import bench.tritonbench_g.chunk_gla_simple.Proofs.Sources.Memory

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.Execution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
open Memory
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

@[simp] theorem inputs_setReg (r : Regions) (s : BlockState) (name : RegName)
    (dt : TileDType) (sh : TileShape) (v : Tile dt sh) :
    inputs r (s.setReg name dt sh v) = inputs r s := rfl

theorem inputs_of_mem (r : Regions) (s t : BlockState) (h : s.mem = t.mem) : inputs r s = inputs r t := by
  unfold inputs
  congr 1 <;> funext off <;> simp only [BlockState.readMem,h]

structure Holds (p : Layout) (r : Regions) (s0 : BlockState) (n : Nat) (u : BlockState) : Prop where
  limit : n ≤ numBlocks p
  memory : u.mem = s0.mem
  padding : u.undef = s0.undef
  pids : u.pids = s0.pids
  valueReg : u.regs .int [] "i_v" = some (Tile.scalar (s0.pids 0 : Int))
  timeReg : u.regs .int [] "i_t" = some (Tile.scalar (s0.pids 1 : Int))
  headReg : u.regs .int [] "i_bh" = some (Tile.scalar (s0.pids 2 : Int))
  maskReg : u.regs .bool [p.BT,p.BT] "m_s" = some (maskTile p)
  outputReg : u.regs .real [p.BT,p.BV] "b_o" = some (carryO p (launch s0) (inputs r s0) n)
  scoreReg : u.regs .real [p.BT,p.BT] "b_s" = some (carryS p (launch s0) (inputs r s0) n)
  rowsReg : u.regs .int [p.BT] "rows" = some (vector p.BT (s0.pids 1*p.BT))
  colsReg : u.regs .int [p.BV] "cols" = some (vector p.BV (s0.pids 0*p.BV))

def start (p : Layout) (s : BlockState) : BlockState :=
  (((((((((((((((s).setReg "i_v" .int [] (Tile.scalar (s.pids 0 : Int))).setReg "i_t" .int [] (Tile.scalar (s.pids 1 : Int))).setReg "i_bh" .int [] (Tile.scalar (s.pids 2 : Int))).setReg "s_k_h" .int [] (Tile.scalar (p.keyHead : Int))).setReg "s_k_t" .int [] (Tile.scalar (p.keyTime : Int))).setReg "s_v_h" .int [] (Tile.scalar (p.valueHead : Int))).setReg "s_v_t" .int [] (Tile.scalar (p.valueTime : Int))).setReg "s_h_h" .int [] (Tile.scalar (p.historyHead : Int))).setReg "s_h_t" .int [] (Tile.scalar (p.historyKey : Int))).setReg "o_i" .nat [p.BT] (Tile.vec fun i => i.val)).setReg "m_s" .bool [p.BT,p.BT] (maskTile p)).setReg "b_o" .real [p.BT,p.BV] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BV])).setReg "b_s" .real [p.BT,p.BT] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT])).setReg "rows" .int [p.BT] (vector p.BT (s.pids 1*p.BT))).setReg "cols" .int [p.BV] (vector p.BV (s.pids 0*p.BV))

theorem setup_executes (p : Layout) (s : BlockState) :
    stepStmts (Program.setup p) s = some (start p s) := by
  unfold Program.setup
  simp only [Real.setup,List.drop_succ_cons,List.drop_zero,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.programId 0)) _ =
    some (Tile.scalar (s.pids 0 : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.programId 1)) _ =
    some (Tile.scalar (s.pids 1 : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.programId 2)) _ =
    some (Tile.scalar (s.pids 2 : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.constNat p.keyHead)) _ =
    some (Tile.scalar (p.keyHead : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.constNat p.keyTime)) _ =
    some (Tile.scalar (p.keyTime : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.constNat p.valueHead)) _ =
    some (Tile.scalar (p.valueHead : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.constNat p.valueTime)) _ =
    some (Tile.scalar (p.valueTime : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.constNat p.historyHead)) _ =
    some (Tile.scalar (p.historyHead : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.castNatToInt (.constNat p.historyKey)) _ =
    some (Tile.scalar (p.historyKey : Int)) from by simp only [evalOp,Option.bind_some,Option.bind_eq_bind]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ =
    some (Tile.vec fun i => i.val) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msGe_op_eval _ p.BT (by simp) (by simp) "o_i" (Tile.vec fun i => i.val) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BV] (.const 0)) _ =
    some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BV]) from by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BT] (.const 0)) _ =
    some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT]) from by simp [evalOp]))]
  unfold Program.addresses
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (programIndices_eval _ "i_t" p.BT (s.pids 1) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (programIndices_eval _ "i_v" p.BV (s.pids 0) (by simp)))]
  rw [stepStmts.nil]
  rfl

theorem start_holds (p : Layout) (r : Regions) (s : BlockState) : Holds p r s 0 (start p s) := by
  constructor
  · exact Nat.zero_le _
  · rfl
  · rfl
  · rfl
  all_goals simp [start,carryO,carryS,Tile.mat,qhSum,qkSum]

theorem loop_step (p : Layout) (r : Regions) (s0 : BlockState) (n : Nat) (u : BlockState)
    (hn : n < numBlocks p) (hi : Holds p r s0 n u) :
    ∃ v, stepStmts (Program.iteration p r) (u.setReg "i_k" .int [] (Tile.scalar (n : Int))) = some v ∧
      Holds p r s0 (n+1) v := by
  have ha : inputs r u = inputs r s0 := inputs_of_mem r u s0 hi.memory
  unfold Program.iteration Real.iteration chunkGlaLoopBody
  simp only [List.drop_succ_cons,List.drop_zero,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (indices_eval _ "i_k" p.BK n (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (queryPtr_eval p (launch s0) r n _ (by simp [launch,hi.headReg]) (by simp [launch,hi.rowsReg]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (keyPtr_eval p (launch s0) r n _ (by simp [launch,hi.headReg]) (by simp [launch,hi.rowsReg]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (historyPtr_eval p (launch s0) r n _ (by simp [launch,hi.headReg]) (by simp [launch,hi.timeReg])
      (by simp [launch,hi.colsReg]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (queryLoad_eval p (launch s0) r n _ (by simp) (by simp [launch,hi.rowsReg]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (keyLoad_eval p (launch s0) r n _ (by simp) (by simp [launch,hi.rowsReg]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (historyLoad_eval p (launch s0) r n _ (by simp) (by simp [launch,hi.colsReg]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ p.BT p.BK p.BV "b_o" "b_q" "b_h"
      (carryO p (launch s0) (inputs r s0) n) (queryTile p (launch s0) (inputs r s0) n)
      (historyTile p (launch s0) (inputs r s0) n)
      (by simp [hi.outputReg]) (by simp [ha]) (by simp [ha])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ p.BT p.BK p.BT "b_s" "b_q" "b_k"
      (carryS p (launch s0) (inputs r s0) n) (queryTile p (launch s0) (inputs r s0) n)
      (keyTile p (launch s0) (inputs r s0) n)
      (by simp [hi.scoreReg]) (by simp [ha]) (by simp [ha]))),stepStmts.nil]
  refine ⟨_,rfl,⟨Nat.succ_le_of_lt hn,hi.memory,hi.padding,hi.pids,?_,?_,?_,?_,?_,?_,?_,?_⟩⟩
  all_goals simp [hi.valueReg,hi.timeReg,hi.headReg,hi.maskReg,hi.rowsReg,hi.colsReg,
    output_update,score_update]

private theorem loop_from (p : Layout) (r : Regions) (s0 u : BlockState)
    (n : Nat) (hi : Holds p r s0 n u) :
    ∃ v, stepForRangeIntAux "i_k" (n : Int) (numBlocks p : Int) 1 (Program.iteration p r) u = some v ∧
      Holds p r s0 (numBlocks p) v := by
  by_cases hn : n < numBlocks p
  · obtain ⟨next,he,hi'⟩ := loop_step p r s0 n u hn hi
    obtain ⟨v,hv,hfinal⟩ := loop_from p r s0 next (n+1) hi'
    refine ⟨v,?_,hfinal⟩
    rw [stepForRangeIntAux.step_lt (by decide) (by exact_mod_cast hn),he,Option.bind_some]
    simpa only [← Int.natCast_add] using hv
  · have hn' : n = numBlocks p := by have := hi.limit; omega
    subst n
    exact ⟨u,stepForRangeIntAux.step_ge (by decide) (by omega),hi⟩
termination_by numBlocks p-n

theorem loop_executes (p : Layout) (r : Regions) (s0 u : BlockState) (hi : Holds p r s0 0 u) :
    ∃ v, stepStmt (Program.loop p r) u = some v ∧ Holds p r s0 (numBlocks p) v := by
  obtain ⟨v,hv,hi'⟩ := loop_from p r s0 u 0 hi
  refine ⟨v,?_,hi'⟩
  simpa only [Program.loop,stepForRangeIntAux.forRangeInt_unfold,Real.tripCount,
    evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,numBlocks] using hv

theorem tail_executes (p : Layout) (r : Regions) (scale : ℝ) (s0 u : BlockState)
    (hi : Holds p r s0 (numBlocks p) u) :
    ∃ v, stepStmts (Program.tail p r scale) u = some v ∧ v.mem = s0.mem ∧
      v.regs .ptr [p.BT,p.BV] "p_o" = some (valuePointer p (launch s0) r.output) ∧
      v.regs .real [p.BT,p.BV] "b_o" = some (resultTile p (launch s0) (inputs r s0) scale) ∧
      v.regs .int [p.BT] "rows" = some (vector p.BT (s0.pids 1*p.BT)) ∧
      v.regs .int [p.BV] "cols" = some (vector p.BV (s0.pids 0*p.BV)) := by
  have ha : inputs r u = inputs r s0 := inputs_of_mem r u s0 hi.memory
  unfold Program.tail Real.tail chunkGlaFront
  simp only [List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (gatePtr_eval p (launch s0) r _ (by simp [launch,hi.headReg]) (by simp [launch,hi.rowsReg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (gateLoad_eval p (launch s0) r _ (by simp) (by simp [launch,hi.rowsReg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mulExpCol_op_eval _ p.BT p.BV (by simp) "b_o" "b_g"
      (carryO p (launch s0) (inputs r s0) (numBlocks p)) (gateTile p (launch s0) (inputs r s0))
      (by simp [hi.outputReg]) (by simp [ha])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mulExpOuter_op_eval _ p.BT (by simp) (by simp) "b_s" "b_g"
      (carryS p (launch s0) (inputs r s0) (numBlocks p)) (gateTile p (launch s0) (inputs r s0))
      (by simp [hi.scoreReg]) (by simp [ha])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (whereMask_op_eval _ p.BT "m_s" "b_s" (maskTile p) (decayedS p (launch s0) (inputs r s0))
      (by simp [hi.maskReg]) (by simp [decayedS])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (valuePtr_eval p (launch s0) r.value _ (by simp [launch,hi.headReg])
      (by simp [launch,hi.rowsReg]) (by simp [launch,hi.colsReg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (valueLoad_eval p (launch s0) r _ (by simp) (by simp [launch,hi.rowsReg]) (by simp [launch,hi.colsReg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accScale_op_eval _ p.BT p.BT p.BV scale "b_o" "b_s" "b_v"
      (decayedO p (launch s0) (inputs r s0)) (maskedS p (launch s0) (inputs r s0))
      (valueTile p (launch s0) (inputs r s0))
      (by simp [decayedO]) (by simp [maskedS]) (by simp [ha])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (valuePtr_eval p (launch s0) r.output _ (by simp [launch,hi.headReg])
      (by simp [launch,hi.rowsReg]) (by simp [launch,hi.colsReg]))),stepStmts.nil]
  refine ⟨_,rfl,hi.memory,?_,?_,?_,?_⟩ <;> simp [resultTile,hi.rowsReg,hi.colsReg]

theorem store_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hp : s.regs .ptr [p.BT,p.BV] "p_o" = some (valuePointer p l r.output))
    (ho : s.regs .real [p.BT,p.BV] "b_o" = some (resultTile p l a scale))
    (hr : s.regs .int [p.BT] "rows" = some (vector p.BT (l.chunk*p.BT)))
    (hc : s.regs .int [p.BV] "cols" = some (vector p.BV (l.value*p.BV))) :
    stepStmt (Program.store p) s = some (scatter p l r a scale s) := by
  simp only [Program.store,stepStmt,evalOp_ref,hp,ho,
    mask2_eval s "rows" "cols" p.BT p.BV p.T p.V _ _ hr hc,
    Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  unfold scatter
  congr 1
  funext u i
  obtain ⟨i,j,⟨⟩⟩ := i
  simp only [BlockState.writeMemTyped_real,decide_eq_true_eq,if_true]
  erw [result_value]
  rfl

theorem region_executes (p : Layout) (r : Regions) (scale : ℝ) (s : BlockState)
    (hinj : NoCollision p (launch s)) :
    ∃ t, exec (Program.typed p r scale).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [p.BT,p.BV], active p (launch s) i.1.val i.2.1.val →
        t.readMem r.output (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r) s t := by
  obtain ⟨v,hl,hv⟩ := loop_executes p r s (start p s) (start_holds p r s)
  obtain ⟨u,ht,hm,hptr,ho,hr,hc⟩ := tail_executes p r scale s v hv
  have he : exec (Program.typed p r scale).eraseDType.toAlgKernel s =
      some (scatter p (launch s) r (inputs r s) scale u) := by
    show stepStmts _ _ = _
    rw [Program.source_body,Program.body]
    simp only [List.append_assoc]
    rw [stepStmts.append_some (setup_executes p s)]
    have hlone : stepStmts [Program.loop p r] (start p s) = some v := by
      rw [stepStmts.cons_some hl,stepStmts.nil]
    rw [stepStmts.append_some hlone,stepStmts.append_some ht,
      stepStmts.cons_some (store_executes p (launch s) r (inputs r s) scale u hptr ho hr hc),stepStmts.nil]
  refine ⟨_,he,fun i hi => scatter_readback p (launch s) r (inputs r s) scale u hinj i hi,?_⟩
  intro region off hn
  simpa only [hm] using scatter_frame p (launch s) r (inputs r s) scale u region off hn

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.Execution
