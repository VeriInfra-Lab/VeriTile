import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealGateSafety
import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealLoop
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.LoopSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
open RealHAccessSafety (load2_safe store2_safe)
open RebasedAccess VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem prelude_safe (p : Parameters) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (f : StateValue p.toLayout) (hfull : Full p.toLayout l c) (bounds : RegionBounds) (u : BlockState) (hb : StepBounds p.toLayout l r c p.gateK bounds)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (hit : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f)) (hstr : StrideRegs p u) :
    Stmt.TraceSafeList bounds (iterationPrefix p r) u := by
  rw [iteration_prefix_ast]
  simp only [RealHProgram.iterationPrelude,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_pointer_eval p l r.key c u hih hik hit hstr))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_pointer_eval p l r.value c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit]) (by simpa [StrideRegs] using hstr)))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (history_pointer_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit]) (by simpa [StrideRegs] using hstr)))]
  refine ⟨by simp [Stmt.TraceSafe,historyPtr,integer,reg,natReg,add,mul,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (storeCarry p.toLayout :: _) (Step.pointers p l r c u)
  erw [TraceProofs.cons_of_run_iff (RebasedStore.store_eval p.toLayout l r.history (RealHStep.historyBase p.toLayout l c)
    p.historyKey p.historyValue f "p_h" _ hfull.key hfull.value (by simp [Step.pointers,hh]) (by simp [Step.pointers,RebasedStore.shifted,historyPointer,RealHStep.historyBase]))]
  refine ⟨?_,?_⟩
  · exact store2_safe r.history (l.head*p.historyHead+c*p.K*p.V+l.key*p.BK*p.historyKey+l.value*p.BV*p.historyValue) p.BK p.BV p.BK p.BV
      p.historyKey p.historyValue 0 0 "p_h" "b_h" bounds _ hb.history
      (by simp [Step.pointers,historyPointer,matrix])
  apply TraceProofs.assign_cons (load2_safe r.key (l.head*p.keyHead+l.key*p.BK*p.keyFeature+c*p.BT*p.keyTime) p.BK p.BT p.BK p.BT
    p.keyFeature p.keyTime 0 0 "p_k" bounds _ hb.key
    (by simp [RealHStore.scatter,Step.pointers,keyPointer,matrix]))
  intro key
  apply TraceProofs.assign_cons (load2_safe r.value (l.head*p.valueHead+c*p.BT*p.valueTime+l.value*p.BV*p.valueFeature) p.BT p.BV p.BT p.BV
    p.valueTime p.valueFeature 0 0 "p_v" bounds _ hb.value
    (by simp [RealHStore.scatter,Step.pointers,valuePointer,matrix]))
  intro value
  exact Stmt.TraceSafeList.nil_intro

theorem iteration_safe (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState)
    (hfull : Full p.toLayout l c) (hs : RealHStep.Separate r)
    (bounds : RegionBounds) (hb : StepBounds p.toLayout l r c p.gateK bounds)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (hit : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f)) (hstr : StrideRegs p u) :
    Stmt.TraceSafeList bounds (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.iteration p r) u := by
  have he := Step.prefix_executes p l r a c f u hs hfull hik hiv hih hit hk hv hh hstr
  have hp := prelude_safe p l r c p.gateK f hfull bounds u hb hik hiv hih hit hh hstr
  rw [iteration_decomposition]
  apply TraceProofs.append_safe _ u hp
  intro v hv
  rw [he] at hv
  cases hv
  apply Stmt.TraceSafeList.cons_intro (GateSafety.update_safe p l r c p.gateK p.swapped bounds _ hb.gate
    (by simp [Step.prepared,Step.stored,RealHStore.scatter,Step.pointers,hik])
    (by simp [Step.prepared,Step.stored,RealHStore.scatter,Step.pointers,hiv])
    (by simp [Step.prepared,Step.stored,RealHStore.scatter,Step.pointers,hih])
    (by simp [Step.prepared,Step.stored,RealHStore.scatter,Step.pointers,hit])
    (by simpa [StrideRegs,Step.prepared,Step.stored,RealHStore.scatter,Step.pointers] using hstr) hfull.positiveTime)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem auxiliary_safe (initial : BlockState) (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
    (hfull : ∀ c,c<p.NT→Full p.toLayout l c)
    (hs : RealHStep.Separate r) (hinj : RealHLoop.HistoryNoCollision p.toLayout l)
    (bounds : RegionBounds) (hb : LoopBounds p.toLayout l r p.gateK bounds)
    (c : Nat) (u : BlockState) (h : Loop.Holds initial p l r a c u) :
    Stmt.forRangeIntTraceSafe bounds "i_t" (c : Int) (p.NT : Int) 1
      (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.iteration p r) u := by
  rw [Stmt.forRangeIntTraceSafe]
  simp only [Nat.one_ne_zero,↓reduceIte]
  by_cases hc:c<p.NT
  · have hci : (c : Int)<(p.NT : Int):=by exact_mod_cast hc
    simp only [hci,↓reduceIte]
    have hi:=iteration_safe p l r a c (expected p.toLayout l a p.gateK p.hasSeed c)
      (u.setReg "i_t" .int [] (Tile.scalar (c : Int))) (hfull c hc) hs bounds (hb c hc)
      (by simpa using h.key) (by simpa using h.value) (by simpa using h.head) (by simp)
      (by simpa using h.inputKey) (by simpa using h.inputValue) (by simpa using h.carry)
      (by simpa [StrideRegs] using h.strides)
    refine ⟨hi,?_⟩
    obtain ⟨v,he,hv⟩:=Loop.iteration initial p l r a hfull hs hinj c u hc h
    rw [he]
    simpa only [Int.natCast_add,Int.natCast_one] using
      auxiliary_safe initial p l r a hfull hs hinj bounds hb (c+1) v hv
  · have hci : ¬(c : Int)<(p.NT : Int):=by exact_mod_cast hc
    simp only [hci,↓reduceIte]
termination_by p.NT-c

theorem loop_safe (initial : BlockState) (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
    (hfull : ∀ c,c<p.NT→Full p.toLayout l c)
    (hs : RealHStep.Separate r) (hinj : RealHLoop.HistoryNoCollision p.toLayout l)
    (bounds : RegionBounds) (hb : LoopBounds p.toLayout l r p.gateK bounds) (u : BlockState)
    (h : Loop.Holds initial p l r a 0 u) :
    Stmt.TraceSafe bounds (loopStatement p r) u := by
  rw [iteration_structure,Stmt.TraceSafe]
  simpa only [Op.SafeAt,evalOp,Tile.scalar,Option.bind_some,Option.bind_eq_bind,
    Option.pure_def,Int.ofNat_eq_natCast,true_and] using
    auxiliary_safe initial p l r a hfull hs hinj bounds hb 0 u h

#axiomsClean prelude_safe
#axiomsClean iteration_safe
#axiomsClean auxiliary_safe
#axiomsClean loop_safe
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.LoopSafety
