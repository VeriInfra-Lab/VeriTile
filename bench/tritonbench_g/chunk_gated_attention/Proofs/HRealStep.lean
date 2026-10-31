import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealGate
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedStore
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHStep
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.Step
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHAlgebra RebasedAccess
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
abbrev historyBase := RealHStep.historyBase

structure Result (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u v : BlockState) : Prop where
  keyIndex : v.regs .int [] "i_k"=some (Tile.scalar (l.key : Int))
  valueIndex : v.regs .int [] "i_v"=some (Tile.scalar (l.value : Int))
  headIndex : v.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int))
  timeIndex : v.regs .int [] "i_t"=some (Tile.scalar (c : Int))
  carry : v.regs .real [p.BK,p.BV] "b_h"=some (stateTile p.toLayout
    (advance (RealHTiles.loaded p.toLayout l r a (fun _ _=>0) c) p.gateK f))
  inputKey : ∀ off,v.readMem r.key off=a.key off
  inputValue : ∀ off,v.readMem r.value off=a.value off
  inputGate : ∀ off,v.readMem r.gate off=a.gate off
  history : ∀ (e : Fin p.BK) (j : Fin p.BV),RealHStore.active p.toLayout l e.val j.val→
    v.readMem r.history (RealHStore.address p.toLayout l (RealHStep.historyBase p.toLayout l c)
      p.historyKey p.historyValue e.val j.val)=f e j
  frame : BlockState.WriteWithin (RealHStore.footprint p.toLayout l r.history
    (RealHStep.historyBase p.toLayout l c) p.historyKey p.historyValue) u v
  pids : v.pids=u.pids
  undef : v.undef=u.undef
  strides : StrideRegs p v


def pointers (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (c : Nat) (u : BlockState) : BlockState :=
  ((u.setReg "p_k" .blockPtr [p.BK,p.BT] (keyPointer p.toLayout l r.key c)).setReg
    "p_v" .blockPtr [p.BT,p.BV] (valuePointer p.toLayout l r.value c)).setReg
    "p_h" .blockPtr [p.BK,p.BV] (historyPointer p.toLayout l r.history c)

def stored (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (c : Nat) (f : StateValue p.toLayout) (u : BlockState) : BlockState :=
  RealHStore.scatter p.toLayout l r.history (historyBase p.toLayout l c) p.historyKey p.historyValue f (pointers p l r c u)

def prepared (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState) : BlockState :=
  let x := RebasedAccess.loaded p.toLayout l a c
  ((stored p l r c f u).setReg "b_k" .real [p.BK,p.BT] (Tile.mat (fun e t => some (x.key e t)))).setReg
    "b_v" .real [p.BT,p.BV] (Tile.mat (fun t j => some (x.value t j)))

theorem stored_other (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (c : Nat)
    (f : StateValue p.toLayout) (u : BlockState) (region : RegionName) (h : region ≠ r.history) :
    ∀ off, (stored p l r c f u).readMem region off = u.readMem region off := by
  intro off
  unfold stored RealHStore.scatter
  rw [BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ h]
  simp [pointers]

theorem prefix_executes (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat) (f : StateValue p.toLayout) (u : BlockState)
    (hs : RealHStep.Separate r) (hfull : RebasedAccess.Full p.toLayout l c)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (hit : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f)) (hstr : StrideRegs p u) :
    stepStmts (iterationPrefix p r) u = some (prepared p l r a c f u) := by
  rw [iteration_prefix_ast]
  simp only [RealHProgram.iterationPrelude, List.drop, List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_pointer_eval p l r.key c u hih hik hit hstr))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_pointer_eval p l r.value c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit]) (by simpa [StrideRegs] using hstr)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (history_pointer_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit]) (by simpa [StrideRegs] using hstr)))]
  change stepStmts (storeCarry p.toLayout :: _) (pointers p l r c u) = _
  erw [stepStmts.cons_some (RebasedStore.store_eval p.toLayout l r.history (historyBase p.toLayout l c)
    p.historyKey p.historyValue f "p_h" _ hfull.key hfull.value (by simp [pointers,hh]) (by simp [pointers,RebasedStore.shifted,historyPointer,RealHStep.historyBase]))]
  change stepStmts (.assign .real [p.BK,p.BT] "b_k" _ :: _) (stored p l r c f u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p.toLayout l r.key a.key c "p_k" _
    (fun off => (stored_other p l r c f u r.key hs.key off).trans (hk off))
    (by simp [stored,RealHStore.scatter,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p.toLayout l r.value a.value c "p_v" _
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

theorem executes (p : Parameters) (l : Launch) (r : RealHProgram.Regions) (a : Inputs)
    (c : Nat)
    (f : StateValue p.toLayout) (u : BlockState) (hs : RealHStep.Separate r) (hfull : RebasedAccess.Full p.toLayout l c)
    (hinj : RealHStore.NoCollision p.toLayout l (historyBase p.toLayout l c) p.historyKey p.historyValue)
    (hik : u.regs .int [] "i_k" = some (Tile.scalar (l.key : Int)))
    (hiv : u.regs .int [] "i_v" = some (Tile.scalar (l.value : Int)))
    (hih : u.regs .int [] "i_bh" = some (Tile.scalar (l.head : Int)))
    (hit : u.regs .int [] "i_t" = some (Tile.scalar (c : Int)))
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p.toLayout f)) (hstr : StrideRegs p u) :
    ∃ v, stepStmts (iteration p r) u = some v ∧ Result p l r a c f u v := by
  let pre := prepared p l r a c f u
  let x := RebasedAccess.loaded p.toLayout l a c
  let mid := Gate.output p l r a c p.gateK f pre
  let v := mid.setReg "b_h" .real [p.BK,p.BV] (stateTile p.toLayout (advance x p.gateK f))
  have hp := prefix_executes p l r a c f u hs hfull hik hiv hih hit hk hv hh hstr
  have heg := Gate.executes p l r a c p.gateK f pre
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hik])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hiv])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hih])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hit])
    (by simp [pre,prepared,stored,RealHStore.scatter,pointers,hh])
    (by simp [pre,prepared]) (by simp [pre,prepared])
    (by simpa [pre,prepared] using (fun off => (stored_other p l r c f u r.gate hs.gate off).trans (hg off)))
    (by simpa [StrideRegs,pre,prepared,stored,RealHStore.scatter,pointers] using hstr) hfull.positiveTime
  have hm := Gate.result p l r a c p.gateK f pre
    (by simp [pre,prepared]) (by simp [pre,prepared])
  have headd := update_eval p.toLayout x p.gateK p.swapped f mid hm.carry hm.key hm.value
  have hmem : v.mem = (stored p l r c f u).mem := by
    change mid.mem = (stored p l r c f u).mem
    exact hm.mem
  refine ⟨v,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [iteration_decomposition,stepStmts.append_some hp]
    exact branch_continuation p.gateK _ _ _ pre mid v heg (stepStmt_assign_eq_some headd)
  · simp [v,mid,Gate.integers,pre,prepared,stored,RealHStore.scatter,pointers,hik]
  · simp [v,mid,Gate.integers,pre,prepared,stored,RealHStore.scatter,pointers,hiv]
  · simp [v,mid,Gate.integers,pre,prepared,stored,RealHStore.scatter,pointers,hih]
  · simp [v,mid,Gate.integers,pre,prepared,stored,RealHStore.scatter,pointers,hit]
  · simp only [v,BlockState.setReg_same]
    rw [← RebasedAccess.loaded_eq p.toLayout l r a (fun _ _ => 0) c hfull]
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
    exact RealHStore.readback p.toLayout l r.history (historyBase p.toLayout l c) p.historyKey p.historyValue f _ hinj e j ha
  · intro region off hn
    rw [hmem]
    exact RealHStore.frame p.toLayout l r.history (historyBase p.toLayout l c) p.historyKey p.historyValue f (pointers p l r c u) region off hn
  · change mid.pids = u.pids
    rw [hm.pids]
    simp [pre,prepared,stored,RealHStore.scatter,BlockState.foldl_writeMem_prop_masked_pids,pointers]
  · change mid.undef = u.undef
    rw [hm.undef]
    change (stored p l r c f u).undef = _
    rw [stored,RealHStore.scatter_undef]
    rfl

  · simpa [StrideRegs,v,mid,Gate.integers,pre,prepared,stored,RealHStore.scatter,pointers] using hstr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.Step
