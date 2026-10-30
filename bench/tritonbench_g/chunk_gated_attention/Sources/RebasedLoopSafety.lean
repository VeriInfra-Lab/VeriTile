import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedGateSafety

/-! Complete iteration and loop access safety. Register addresses follow the
actual execution, including the history store before K/V loads. The previously
proved loop invariant supplies every reached state; no whole-program execution
or output equation is assumed as a precondition. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedLoopSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles
open RealHAccessSafety (load2_safe store2_safe)
open RebasedAccess RebasedAccessSafety
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) : Prop where
  key : KeyBounds p l r.key c bounds
  value : ValueBounds p l r.value c bounds
  history : HistoryBounds p l r.history c bounds
  gate : RebasedGateSafety.Bounds p l r c gateK bounds

def Bounds (p : Layout) (l : Launch) (r : Regions) (gateK : Bool) (bounds : RegionBounds) : Prop :=
  ∀ c, c < p.NT → StepBounds p l r c gateK bounds

theorem prelude_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (f : StateValue p) (hfull : Full p l c) (bounds : RegionBounds) (u : BlockState) (hb : StepBounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (RebasedProgram.prelude p r) u := by
  simp only [RebasedProgram.prelude,RealHProgram.iterationPrelude,List.drop,List.cons_append,List.nil_append]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.key c u hih hik hit))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.keyPtr,RebasedProgram.local2,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.value c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.valuePtr,RebasedProgram.local2,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (history_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe,RebasedProgram.historyPtr,RebasedProgram.local2,times,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (storeCarry p :: _) (RebasedStep.pointers p l r c u)
  erw [TraceProofs.cons_of_run_iff (RebasedStore.store_eval p l r.history (RealHStep.historyBase p l c)
    p.historyKey p.historyValue f "p_h" _ hfull.key hfull.value (by simp [RebasedStep.pointers,hh]) (by simp [RebasedStep.pointers,RebasedStore.shifted,historyPointer,RealHStep.historyBase]))]
  refine ⟨?_,?_⟩
  · exact store2_safe r.history (l.head*p.historyHead+c*p.K*p.V+l.key*p.BK*p.historyKey+l.value*p.BV*p.historyValue) p.BK p.BV p.BK p.BV
      p.historyKey p.historyValue 0 0 "p_h" "b_h" bounds _ hb.history
      (by simp [RebasedStep.pointers,historyPointer,matrix])
  apply TraceProofs.assign_cons (load2_safe r.key (l.head*p.keyHead+l.key*p.BK*p.keyFeature+c*p.BT*p.keyTime) p.BK p.BT p.BK p.BT
    p.keyFeature p.keyTime 0 0 "p_k" bounds _ hb.key
    (by simp [RealHStore.scatter,RebasedStep.pointers,keyPointer,matrix]))
  intro key
  apply TraceProofs.assign_cons (load2_safe r.value (l.head*p.valueHead+c*p.BT*p.valueTime+l.value*p.BV*p.valueFeature) p.BT p.BV p.BT p.BV
    p.valueTime p.valueFeature 0 0 "p_v" bounds _ hb.value
    (by simp [RealHStore.scatter,RebasedStep.pointers,valuePointer,matrix]))
  intro value
  exact Stmt.TraceSafeList.nil_intro

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (gateK swapped : Bool) (f : StateValue p) (u : BlockState)
    (hfull : Full p l c) (hs : RealHStep.Separate r)
    (bounds : RegionBounds) (hb : StepBounds p l r c gateK bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (RebasedProgram.iteration p r gateK swapped) u := by
  have he := RebasedStep.prefix_executes p l r a c f u hs hfull hik hiv hih hit hk hv hh
  have hp := prelude_safe p l r c gateK f hfull bounds u hb hik hiv hih hit hh
  rw [RebasedProgram.iteration]
  apply TraceProofs.append_safe _ u hp
  intro v hv
  rw [he] at hv
  cases hv
  apply Stmt.TraceSafeList.cons_intro (RebasedGateSafety.update_safe p l r c gateK swapped bounds _ hb.gate
    (by simp [RebasedStep.prepared,RebasedStep.stored,RealHStore.scatter,RebasedStep.pointers,hik])
    (by simp [RebasedStep.prepared,RebasedStep.stored,RealHStore.scatter,RebasedStep.pointers,hiv])
    (by simp [RebasedStep.prepared,RebasedStep.stored,RealHStore.scatter,RebasedStep.pointers,hih])
    (by simp [RebasedStep.prepared,RebasedStep.stored,RealHStore.scatter,RebasedStep.pointers,hit]))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed swapped : Bool)
    (hfull : ∀ c, c < p.NT → Full p l c)
    (hs : RealHStep.Separate r) (hinj : RealHLoop.HistoryNoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l r gateK bounds) (u : BlockState)
    (h : RealHLoop.Holds initial p l r a gateK hasSeed 0 u) :
    Stmt.TraceSafe bounds (RebasedLoop.statement p r gateK swapped) u := by
  rw [RebasedLoop.statement,Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := RealHLoop.Holds initial p l r a gateK hasSeed)
  · intro c v hc hv
    apply iteration_safe p l r a c gateK swapped (expected p l a gateK hasSeed c)
      (v.setReg "i_t" .nat [] (Tile.scalar c)) (hfull c hc) hs bounds (hb c hc)
    · simpa using hv.key
    · simpa using hv.value
    · simpa using hv.head
    · simp
    · simpa using hv.inputKey
    · simpa using hv.inputValue
    · simpa using hv.carry
  · intro c v w hc hv he
    obtain ⟨w',he',hw⟩ := RebasedLoop.iteration initial p l r a gateK hasSeed swapped hfull hs hinj c v hc hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedLoopSafety
