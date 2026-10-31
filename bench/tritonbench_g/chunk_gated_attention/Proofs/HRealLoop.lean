import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealStep
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHLoop
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.Loop
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Holds (initial : BlockState) (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (u : BlockState) : Prop where
  count_le : c≤p.NT
  key : u.regs .int [] "i_k"=some (Tile.scalar (l.key : Int))
  value : u.regs .int [] "i_v"=some (Tile.scalar (l.value : Int))
  head : u.regs .int [] "i_bh"=some (Tile.scalar (l.head : Int))
  carry : u.regs .real [p.BK,p.BV] "b_h"=some (stateTile p.toLayout
    (expected p.toLayout l a p.gateK p.hasSeed c))
  inputKey : ∀ off,u.readMem r.key off=a.key off
  inputValue : ∀ off,u.readMem r.value off=a.value off
  inputGate : ∀ off,u.readMem r.gate off=a.gate off
  history : ∀ (q : Nat) (e : Fin p.BK) (j : Fin p.BV),q<c→RealHStore.active p.toLayout l e.val j.val→
    u.readMem r.history (address p.toLayout l q e.val j.val)=expected p.toLayout l a p.gateK p.hasSeed q e j
  frame : BlockState.WriteWithin (footprint p.toLayout l r) initial u
  pids : u.pids=initial.pids
  undef : u.undef=initial.undef
  strides : StrideRegs p u

theorem iteration (initial : BlockState) (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
        (hfull : ∀ c, c < p.NT → RebasedAccess.Full p.toLayout l c)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p.toLayout l)
    (c : Nat) (u : BlockState) (hc : c < p.NT) (h : Holds initial p l r a c u) :
    ∃ v, stepStmts (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.iteration p r) (u.setReg "i_t" .int [] (Tile.scalar (c : Int))) = some v ∧
      Holds initial p l r a (c+1) v := by
  let entry := u.setReg "i_t" .int [] (Tile.scalar (c : Int))
  have hci : RealHStore.NoCollision p.toLayout l (RealHStep.historyBase p.toLayout l c) p.historyKey p.historyValue := by
    intro e f i j he hf heq
    exact (hinj c c e f i j hc hc he hf heq).2
  obtain ⟨v,he,hv⟩ := Step.executes p l r a c
    (expected p.toLayout l a p.gateK p.hasSeed c) entry hs (hfull c hc) hci
    (by simp [entry,h.key]) (by simp [entry,h.value]) (by simp [entry,h.head]) (by simp [entry])
    (by simpa [entry] using h.inputKey) (by simpa [entry] using h.inputValue)
    (by simpa [entry] using h.inputGate) (by simpa [entry] using h.carry)
    (by simpa [entry,StrideRegs] using h.strides)
  refine ⟨v,he,Nat.succ_le_of_lt hc,hv.keyIndex,hv.valueIndex,hv.headIndex,?_,
    hv.inputKey,hv.inputValue,hv.inputGate,?_,?_,?_,?_,hv.strides⟩
  · simpa only [advance_expected] using hv.carry
  · intro q e j hq ha
    by_cases hqc : q = c
    · subst q
      exact hv.history e j ha
    · have hq' : q < c := by omega
      have hf := hv.frame r.history (address p.toLayout l q e.val j.val) (by
        rintro ⟨_,f,i,hi,hoff⟩
        exact hqc (hinj q c e f j i (Nat.lt_trans hq' hc) hc ha hi hoff).1)
      change v.readMem r.history (address p.toLayout l q e.val j.val) = _
      unfold BlockState.readMem
      rw [← hf]
      exact h.history q e j hq' ha
  · intro region off hn
    apply (h.frame region off hn).trans
    apply hv.frame region off
    rintro ⟨hr,e,j,ha,hoff⟩
    exact hn ⟨hr,c,e,j,hc,ha,hoff⟩
  · exact hv.pids.trans h.pids
  · exact hv.undef.trans h.undef

theorem auxiliary (initial : BlockState) (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
    (hfull : ∀ c,c<p.NT→RebasedAccess.Full p.toLayout l c)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p.toLayout l)
    (c : Nat) (u : BlockState) (h : Holds initial p l r a c u) :
    ∃ v,stepForRangeIntAux "i_t" (c : Int) (p.NT : Int) 1
      (VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.iteration p r) u=some v ∧ Holds initial p l r a p.NT v := by
  by_cases hc:c<p.NT
  · obtain ⟨mid,he,hm⟩:=iteration initial p l r a hfull hs hinj c u hc h
    obtain ⟨v,hv,hout⟩:=auxiliary initial p l r a hfull hs hinj (c+1) mid hm
    refine ⟨v,?_,hout⟩
    rw [stepForRangeIntAux.step_lt (by decide) (by exact_mod_cast hc),he,Option.bind_some]
    simpa only [Int.natCast_add,Int.natCast_one] using hv
  · have hn:c=p.NT:=by have :=h.count_le; omega
    subst c
    exact ⟨u,stepForRangeIntAux.step_ge (by decide) (by omega),h⟩
termination_by p.NT-c

theorem executes (initial : BlockState) (p : Parameters) (l : Launch) (r : Regions) (a : Inputs)
    (hfull : ∀ c,c<p.NT→RebasedAccess.Full p.toLayout l c)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p.toLayout l)
    (u : BlockState) (h : Holds initial p l r a 0 u) :
    ∃ v,stepStmt (loopStatement p r) u=some v ∧ Holds initial p l r a p.NT v := by
  obtain ⟨v,he,hv⟩:=auxiliary initial p l r a hfull hs hinj 0 u h
  refine ⟨v,?_,hv⟩
  rw [iteration_structure,stepForRangeIntAux.forRangeInt_unfold]
  simpa only [evalOp,Tile.scalar,Option.bind_some,Option.bind_eq_bind,Option.pure_def,
    Int.ofNat_eq_natCast] using he

#axiomsClean iteration
#axiomsClean auxiliary
#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real.Loop
