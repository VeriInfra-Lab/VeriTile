import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedGate
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedStore
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHStep

/-! Complete repaired iteration. Original numerical and scatter/frame results
are reused; the source-local wrapper discharges the new pointer setup. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedStep
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHAlgebra RebasedAccess
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
abbrev historyBase := RealHStep.historyBase

def pointers (p : Layout) (l : Launch) (r : RealHProgram.Regions) (c : Nat) (u : BlockState) : BlockState :=
  ((u.setReg "p_k" .blockPtr [p.BK,p.BT] (keyPointer p l r.key c)).setReg
    "p_v" .blockPtr [p.BT,p.BV] (valuePointer p l r.value c)).setReg
    "p_h" .blockPtr [p.BK,p.BV] (historyPointer p l r.history c)

def stored (p : Layout) (l : Launch) (r : RealHProgram.Regions) (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  RealHStore.scatter p l r.history (historyBase p l c) p.historyKey p.historyValue f (pointers p l r c u)

def prepared (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p) (u : BlockState) : BlockState :=
  let x := RebasedAccess.loaded p l a c
  ((stored p l r c f u).setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun e t => some (x.key e t)))).setReg
    "b_v" .real [p.BT,p.BV] (Tile.mat (fun t j => some (x.value t j)))

theorem stored_other (p : Layout) (l : Launch) (r : RealHProgram.Regions) (c : Nat)
    (f : StateValue p) (u : BlockState) (region : RegionName) (h : region ≠ r.history) :
    ∀ off, (stored p l r c f u).readMem region off = u.readMem region off := by
  intro off
  unfold stored RealHStore.scatter
  rw [BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ h]
  simp [pointers]

theorem prefix_executes (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p) (u : BlockState)
    (hs : RealHStep.Separate r) (hfull : RebasedAccess.Full p l c)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    stepStmts (RebasedProgram.prelude p r) u = some (prepared p l r a c f u) := by
  simp only [RebasedProgram.prelude, RealHProgram.iterationPrelude, List.drop, List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.key c u hih hik hit))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.value c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (history_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit])))]
  change stepStmts (storeCarry p :: _) (pointers p l r c u) = _
  erw [stepStmts.cons_some (RebasedStore.store_eval p l r.history (historyBase p l c)
    p.historyKey p.historyValue f "p_h" _ hfull.key hfull.value (by simp [pointers,hh]) (by simp [pointers,RebasedStore.shifted,historyPointer,RealHStep.historyBase]))]
  change stepStmts (.assign .real [p.BK,p.BT] "b_k" _ :: _) (stored p l r c f u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.key a.key c "p_k" _
    (fun off => (stored_other p l r c f u r.key hs.key off).trans (hk off))
    (by simp [stored,RealHStore.scatter,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.value a.value c "p_v" _
    (by simpa using (fun off => (stored_other p l r c f u r.value hs.value off).trans (hv off)))
    (by simp [stored,RealHStore.scatter,pointers]))),stepStmts.nil]
  rfl

theorem branch_continuation (gate : Bool) (yes no : List Stmt) (next : Stmt)
    (u mid out : BlockState)
    (hb : stepStmt (.ifThenElse (.constBool gate) yes no) u = some mid)
    (hn : stepStmt next mid = some out) :
    stepStmts [.ifThenElse (.constBool gate) (yes ++ [next]) (no ++ [next])] u = some out := by
  cases gate <;>
    simp only [stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,Tile.scalar,
      Bool.false_eq_true,↓reduceIte] at hb
  all_goals simp only [stepStmts,stepStmt,evalOp,Option.bind_eq_bind,Option.bind_some,
    Tile.scalar,Bool.false_eq_true,↓reduceIte]
  all_goals rw [ProgramRewrite.append,hb]
  all_goals simpa only [Option.bind_some,stepStmts,hn] using (rfl : some out = some out)

theorem executes (p : Layout) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (gateK swapped : Bool)
    (f : StateValue p) (u : BlockState) (hs : RealHStep.Separate r) (hfull : RebasedAccess.Full p l c)
    (hinj : RealHStore.NoCollision p l (historyBase p l c) p.historyKey p.historyValue)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p f)) :
    ∃ v, stepStmts (RebasedProgram.iteration p r gateK swapped) u = some v ∧ RealHStep.Result p l r a (fun _ _ => 0) c gateK f u v := by
  let pre := prepared p l r a c f u
  let x := RebasedAccess.loaded p l a c
  let mid := RebasedGate.output p l r a c gateK f pre
  let v := mid.setReg "b_h" .real [p.BK,p.BV] (stateTile p (advance x gateK f))
  have hp := prefix_executes p l r a c f u hs hfull hik hiv hih hit hk hv hh
  have heg := RebasedGate.executes p l r a c gateK f pre
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hik])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hiv])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hih])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hit])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hh])
    (by simp [pre,prepared]) (by simp [pre,prepared])
    (by simpa [pre,prepared] using (fun off => (stored_other p l r c f u r.gate hs.gate off).trans (hg off)))
  have hm := RebasedGate.result p l r a c gateK f pre
    (by simp [pre,prepared]) (by simp [pre,prepared])
  have headd := update_eval p x gateK swapped f mid hm.carry hm.key hm.value
  have hmem : v.mem = (stored p l r c f u).mem := by
    change mid.mem = (stored p l r c f u).mem
    exact hm.mem
  refine ⟨v,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [RebasedProgram.iteration,stepStmts.append_some hp]
    exact branch_continuation gateK _ _ _ pre mid v heg (stepStmt_assign_eq_some headd)
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hik] using hm.keyIndex
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hiv] using hm.valueIndex
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hih] using hm.headIndex
  · simpa [v,pre,prepared,stored,RealHStore.scatter,pointers,hit] using hm.timeIndex
  · simp only [v,BlockState.setReg_same]
    rw [← RebasedAccess.loaded_eq p l r a (fun _ _ => 0) c hfull]
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.key hs.key off).trans (hk off)
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.value hs.value off).trans (hv off)
  · intro off
    unfold BlockState.readMem
    rw [hmem]
    exact (stored_other p l r c f u r.gate hs.gate off).trans (hg off)
  · intro e j ha
    unfold BlockState.readMem
    rw [hmem]
    exact RealHStore.readback p l r.history (historyBase p l c) p.historyKey p.historyValue f _ hinj e j ha
  · intro region off hn
    rw [hmem]
    exact RealHStore.frame p l r.history (historyBase p l c) p.historyKey p.historyValue f (pointers p l r c u) region off hn
  · change mid.pids = u.pids
    rw [hm.pids]
    simp [pre,prepared,stored,RealHStore.scatter,BlockState.foldl_writeMem_prop_masked_pids,pointers]
  · change mid.undef = u.undef
    rw [hm.undef]
    change (stored p l r c f u).undef = _
    rw [stored,RealHStore.scatter_undef]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedStep
