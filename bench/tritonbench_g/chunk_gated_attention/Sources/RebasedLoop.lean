import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedStep
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHLoop

/-! Repaired loop execution uses the unchanged closed-form invariant, history
separation and frame definitions. No zero-undef premise is needed because each
new pointer's entire tile is staged and active. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedLoop
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHLoop
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed swapped : Bool)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p l c)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p l)
    (c : Nat) (u : BlockState) (hc : c < p.NT) (h : Holds initial p l r a gateK hasSeed c u) :
    ∃ v, stepStmts (RebasedProgram.iteration p r gateK swapped) (u.setReg "i_t" .nat [] (Tile.scalar c)) = some v ∧
      Holds initial p l r a gateK hasSeed (c+1) v := by
  let entry := u.setReg "i_t" .nat [] (Tile.scalar c)
  have hci : RealHStore.NoCollision p l (RealHStep.historyBase p l c) p.historyKey p.historyValue := by
    intro e f i j he hf heq
    exact (hinj c c e f i j hc hc he hf heq).2
  obtain ⟨v,he,hv⟩ := RebasedStep.executes p l r a c gateK swapped
    (expected p l a gateK hasSeed c) entry hs (hfull c hc) hci
    (by simp [entry,h.key]) (by simp [entry,h.value]) (by simp [entry,h.head]) (by simp [entry])
    (by simpa [entry] using h.inputKey) (by simpa [entry] using h.inputValue)
    (by simpa [entry] using h.inputGate) (by simpa [entry] using h.carry)
  refine ⟨v,he,Nat.succ_le_of_lt hc,hv.keyIndex,hv.valueIndex,hv.headIndex,?_,
    hv.inputKey,hv.inputValue,hv.inputGate,?_,?_,?_,?_⟩
  · simpa only [advance_expected] using hv.carry
  · intro q e j hq ha
    by_cases hqc : q = c
    · subst q
      exact hv.history e j ha
    · have hq' : q < c := by omega
      have hf := hv.frame r.history (address p l q e.val j.val) (by
        rintro ⟨_,f,i,hi,hoff⟩
        exact hqc (hinj q c e f j i (Nat.lt_trans hq' hc) hc ha hi hoff).1)
      change v.readMem r.history (address p l q e.val j.val) = _
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

def statement (p : Layout) (r : Regions) (gateK swapped : Bool) : Stmt :=
  .forRange "i_t" 0 p.NT 1 (RebasedProgram.iteration p r gateK swapped)

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed swapped : Bool)
    (hfull : ∀ c, c < p.NT → RebasedAccess.Full p l c)
    (hs : RealHStep.Separate r) (hinj : HistoryNoCollision p l)
    (u : BlockState) (h : Holds initial p l r a gateK hasSeed 0 u) :
    ∃ v, stepStmt (statement p r gateK swapped) u = some v ∧ Holds initial p l r a gateK hasSeed p.NT v := by
  obtain ⟨last,v,he,hend,hv⟩ := forRange_inv (P := Holds initial p l r a gateK hasSeed)
    (by decide : 1 ≠ 0) h
    (fun c u hc h => iteration initial p l r a gateK hasSeed swapped hfull hs hinj c u hc h)
  have hlast : last = p.NT := Nat.le_antisymm hv.count_le hend
  subst last
  exact ⟨v,he,hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedLoop
