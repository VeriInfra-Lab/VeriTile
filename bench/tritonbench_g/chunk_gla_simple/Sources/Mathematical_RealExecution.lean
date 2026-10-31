import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_RealMath
import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_RealProgram

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def iteration (p : Layout) (r : Regions) : List Stmt :=
  chunkGlaLoopBody r.query r.key r.value r.history p.keyHead p.keyTime p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV

def tripCount (p : Layout) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1)) (.constNat p.BK)

structure Holds (p : Layout) (r : Regions) (s0 : BlockState) (n : Nat) (u : BlockState) : Prop where
  limit : n ≤ numBlocks p
  memory : u.mem = s0.mem
  padding : u.undef = s0.undef
  pids : u.pids = s0.pids
  valueReg : u.regs .nat [] "i_v" = some (Tile.scalar (s0.pids 0))
  timeReg : u.regs .nat [] "i_t" = some (Tile.scalar (s0.pids 1))
  headReg : u.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2))
  maskReg : u.regs .bool [p.BT,p.BT] "m_s" = some (maskTile p)
  outputReg : u.regs .real [p.BT,p.BV] "b_o" = some (carryO p (launch s0) (inputs r s0) n)
  scoreReg : u.regs .real [p.BT,p.BT] "b_s" = some (carryS p (launch s0) (inputs r s0) n)

theorem loop_step (p : Layout) (r : Regions) (s0 : BlockState)
    (hu : s0.undef = fun _ _ => 0) (n : Nat) (u : BlockState)
    (hn : n < numBlocks p) (hi : Holds p r s0 n u) :
    ∃ v, stepStmts (iteration p r) (u.setReg "i_k" .nat [] (Tile.scalar n)) = some v ∧
      Holds p r s0 (n+1) v := by
  let t := u.setReg "i_k" .nat [] (Tile.scalar n)
  have ht_l : launch t = launch s0 := by simp [launch,t,hi.pids]
  have ht_a : inputs r t = inputs r s0 := by
    change inputs r u = inputs r s0
    unfold inputs
    congr 1 <;> funext off <;> simp only [BlockState.readMem,hi.memory]
  obtain ⟨v,he,hp,hm,hpad,hv,ht,hh,hmask,ho,hs⟩ :=
    chunkGlaLoopBody_steps r.query r.key r.value r.history p.keyHead p.keyTime p.historyHead p.historyKey
      p.T p.K p.V p.BT p.BK p.BV n t (carryO p (launch s0) (inputs r s0) n)
      (carryS p (launch s0) (inputs r s0) n)
      (by simpa [t,hi.pids] using hi.valueReg) (by simpa [t,hi.pids] using hi.timeReg)
      (by simpa [t,hi.pids] using hi.headReg) (by simp [t])
      (by simpa [t] using hi.outputReg) (by simpa [t] using hi.scoreReg)
      (show u.undef = fun _ _ => 0 from hi.padding.trans hu)
  refine ⟨v,he,⟨Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩⟩
  · exact hm.trans hi.memory
  · exact hpad.trans hi.padding
  · exact hp.trans hi.pids
  · simpa [t,hi.pids] using hv
  · simpa [t,hi.pids] using ht
  · simpa [t,hi.pids] using hh
  · simpa [t,hi.maskReg] using hmask
  · rw [ho,query_tile p r t n,history_tile p r t n,ht_l,ht_a,output_update]
  · rw [hs,query_tile p r t n,key_tile p r t n,ht_l,ht_a,score_update]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def setup (p : Layout) : List Stmt :=
  [.assign .nat [] "i_v" (.programId 0),
   .assign .nat [] "i_t" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [p.BT] "o_i" (.arange p.BT),
   .assign .bool [p.BT,p.BT] "m_s"
      (.ge .nat (.consR (.consL .nil))
        (.expandDim ⟨1,by simp⟩ (.ref .nat [p.BT] "o_i"))
        (.expandDim ⟨0,by simp⟩ (.ref .nat [p.BT] "o_i"))),
   .assign .real [p.BT,p.BV] "b_o" (.full [p.BT,p.BV] (.const 0)),
   .assign .real [p.BT,p.BT] "b_s" (.full [p.BT,p.BT] (.const 0))]

def start (p : Layout) (s : BlockState) : BlockState :=
  ((((((s.setReg "i_v" .nat [] (Tile.scalar (s.pids 0))).setReg "i_t" .nat [] (Tile.scalar (s.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg "o_i" .nat [p.BT] (Tile.vec fun i => i.val)).setReg
    "m_s" .bool [p.BT,p.BT] (maskTile p)).setReg
    "b_o" .real [p.BT,p.BV] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BV])).setReg
    "b_s" .real [p.BT,p.BT] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT])

theorem setup_executes (p : Layout) (s : BlockState) :
    stepStmts (setup p) s = some (start p s) := by
  unfold setup
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ =
    some (Tile.vec fun i => i.val) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msGe_op_eval _ p.BT (by simp) (by simp) "o_i" (Tile.vec fun i => i.val) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BV] (.const 0)) _ =
    some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BV]) from by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BT] (.const 0)) _ =
    some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT]) from by simp [evalOp])),stepStmts.nil]
  rfl

theorem start_holds (p : Layout) (r : Regions) (s : BlockState) : Holds p r s 0 (start p s) := by
  constructor
  · exact Nat.zero_le _
  · rfl
  · rfl
  · rfl
  · simp [start]
  · simp [start]
  · simp [start]
  · simp [start]
  · simp [start,carryO,Tile.mat,qhSum]
  · simp [start,carryS,Tile.mat,qkSum]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def tail (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  (chunkGlaFront r.query r.key r.value r.history r.gate r.output
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV scale).drop 8

def outputPointer (p : Layout) (l : Launch) (r : Regions) : Tile .blockPtr [p.BT,p.BV] :=
  ⟨fun _ => BlockPtr.mk r.output (l.head*p.valueHead) [p.T,p.V] [p.BT,p.BV] [p.valueTime,1]
    [l.chunk*p.BT,l.value*p.BV]⟩

theorem tail_executes (p : Layout) (r : Regions) (scale : ℝ) (s0 u : BlockState)
    (hu : s0.undef = fun _ _ => 0) (hi : Holds p r s0 (numBlocks p) u) :
    ∃ v, stepStmts (tail p r scale) u = some v ∧ v.mem = s0.mem ∧ v.undef = s0.undef ∧ v.pids = s0.pids ∧
      v.regs .blockPtr [p.BT,p.BV] "p_o" = some (outputPointer p (launch s0) r) ∧
      v.regs .real [p.BT,p.BV] "b_o" = some (resultTile p (launch s0) (inputs r s0) scale) := by
  have hmem (rg : RegionName) (off : Nat) : u.readMem rg off = s0.readMem rg off := by
    simp only [BlockState.readMem,hi.memory]
  have hpad : u.undef = fun _ _ => 0 := hi.padding.trans hu
  unfold tail chunkGlaFront
  simp only [List.drop_succ_cons,List.drop_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_1d_eval r.gate u _ _ [p.T] [p.BT] [1] (s0.pids 2*p.T) (s0.pids 1*p.BT)
      (mulConst_eval _ "i_bh" _ _ hi.headReg) (mulConst_eval _ "i_t" _ _ hi.timeReg)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_1d_ref r.gate _ "p_g" (s0.pids 2*p.T) p.T p.BT 1 (s0.pids 1*p.BT) (by simp)
      (by simpa using hpad)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mulExpCol_op_eval _ p.BT p.BV (by simp) "b_o" "b_g"
      (carryO p (launch s0) (inputs r s0) (numBlocks p)) (gateTile p (launch s0) (inputs r s0))
      (by simp [hi.outputReg]) (by
        simp only [BlockState.setReg_same]
        apply congrArg some
        apply Tile.ext
        rintro ⟨i,⟨⟩⟩
        simp only [gateTile,gate,gateAddress,row,launch,inputs,Tile.vec,BlockState.setReg_readMem,hmem,mul_one]
        split_ifs <;> rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mulExpOuter_op_eval _ p.BT (by simp) (by simp) "b_s" "b_g"
      (carryS p (launch s0) (inputs r s0) (numBlocks p)) (gateTile p (launch s0) (inputs r s0))
      (by simp [hi.scoreReg]) (by
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide : ("b_g" : RegName) ≠ "b_o"),
          BlockState.setReg_same]
        apply congrArg some
        apply Tile.ext
        rintro ⟨i,⟨⟩⟩
        simp only [gateTile,gate,gateAddress,row,launch,inputs,Tile.vec,BlockState.setReg_readMem,hmem,mul_one]
        split_ifs <;> rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (whereMask_op_eval _ p.BT "m_s" "b_s" (maskTile p) (decayedS p (launch s0) (inputs r s0))
      (by simp [hi.maskReg]) (by simp [decayedS])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.value _ _ _ _ [p.T,p.V] [p.BT,p.BV] [p.valueTime,1] (s0.pids 2*p.valueHead)
      (s0.pids 1*p.BT) (s0.pids 0*p.BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hi.headReg])) (mulConst_eval _ "i_t" _ _ (by simp [hi.timeReg]))
      (mulConst_eval _ "i_v" _ _ (by simp [hi.valueReg]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref r.value _ "p_v" (s0.pids 2*p.valueHead) p.T p.V p.BT p.BV p.valueTime 1
      (s0.pids 1*p.BT) (s0.pids 0*p.BV) (by simp) (by simpa using hpad)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accScale_op_eval _ p.BT p.BT p.BV scale "b_o" "b_s" "b_v"
      (decayedO p (launch s0) (inputs r s0)) (maskedS p (launch s0) (inputs r s0))
      (valueTile p (launch s0) (inputs r s0))
      (by simp [decayedO]) (by simp [maskedS]) (by
        simp only [BlockState.setReg_same]
        apply congrArg some
        apply Tile.ext
        rintro ⟨i,j,⟨⟩⟩
        simp only [valueTile,value,active,outputAddress,row,column,launch,inputs,Tile.mat,
          BlockState.setReg_readMem,hmem,mul_one]
        split_ifs <;> rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.output _ _ _ _ [p.T,p.V] [p.BT,p.BV] [p.valueTime,1] (s0.pids 2*p.valueHead)
      (s0.pids 1*p.BT) (s0.pids 0*p.BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hi.headReg])) (mulConst_eval _ "i_t" _ _ (by simp [hi.timeReg]))
      (mulConst_eval _ "i_v" _ _ (by simp [hi.valueReg])))),stepStmts.nil]
  refine ⟨_,rfl,hi.memory,hi.padding,hi.pids,?_,?_⟩
  · simp [outputPointer,launch]
  · simp [resultTile]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def scatter (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BT,p.BV]).foldl (fun u i =>
    if active p l i.1.val i.2.1.val then
      u.writeMem r.output (outputAddress p l i.1.val i.2.1.val) (expected p l a scale i.1.val i.2.1.val) else u) s

def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ i j : TileIndex [p.BT,p.BV], active p l i.1.val i.2.1.val → active p l j.1.val j.2.1.val →
    outputAddress p l i.1.val i.2.1.val = outputAddress p l j.1.val j.2.1.val → i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = r.output ∧ ∃ i : TileIndex [p.BT,p.BV],
    active p l i.1.val i.2.1.val ∧ cell.2 = outputAddress p l i.1.val i.2.1.val

theorem scatter_readback (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hinj : NoCollision p l) (i : TileIndex [p.BT,p.BV]) (hi : active p l i.1.val i.2.1.val) :
    (scatter p l r a scale s).readMem r.output (outputAddress p l i.1.val i.2.1.val) =
      expected p l a scale i.1.val i.2.1.val := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true s _ _ _ i hi
    (fun j hj he => hinj j i hj hi he)

theorem scatter_frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState) :
    BlockState.WriteWithin (footprint p l r) s (scatter p l r a scale s) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,i,hi,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def totalOp (p : Layout) : Op .real [p.BT,p.BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BT,p.BV] "b_o")
    (.dot (batch := []) (.ref .real [p.BT,p.BT] "b_s") (.ref .real [p.BT,p.BV] "b_v")
      {allowTF32 := some Bool.false})

theorem scaling_commutes (p : Layout) (scale : ℝ) (u : BlockState) :
    stepStmt (.assign .real [p.BT,p.BV] "b_o" (.mul .real .scalarR (totalOp p) (.const scale))) u =
    stepStmt (.assign .real [p.BT,p.BV] "b_o" (.mul .real .scalarL (.const scale) (totalOp p))) u := by
  simp only [stepStmt,evalOp_mul,evalOp_const]
  cases hv : evalOp (totalOp p) u
  · rfl
  rename_i v
  simp only [Option.bind_eq_bind,Option.bind_some]
  congr 2
  apply Tile.ext
  intro i
  cases h : v.data i <;>
    simp [Tile.bop,Tile.scalar,-Tile.scalar_eta,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,
      WithBot.realMul,h,mul_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton Math
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
