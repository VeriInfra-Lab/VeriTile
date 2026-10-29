import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstStep
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Addressing

/-! The actual first backward loop reconstructs h and produces every d_q
from the initial memory snapshot. Its frame preserves all four inputs needed
by the later reverse loop, together with every untouched tagged memory cell. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstLoop
open VeriTile.Triton Primitives MaskedMath BackwardFirstState
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

structure Holds (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : Prop where
  ready : Ready p l initial r hasSeed t u
  qmem : ∀ off, u.readMem r.q off = initial.readMem r.q off
  kmem : ∀ off, u.readMem r.k off = initial.readMem r.k off
  vmem : ∀ off, u.readMem r.v off = initial.readMem r.v off
  domem : ∀ off, u.readMem r.do_ off = initial.readMem r.do_ off
  output : ∀ (k : Nat) (j : Fin p.BK), k < t → activeKey p l j.val →
    u.mem r.dq (keyGradientAddress p l k j.val) = MemCell.real (MaskedMath.dq p l initial r hasSeed k j)
  frame : ∀ (region : RegionName) (off : Nat),
    (∀ (k : Nat) (j : Fin p.BK), k < t → activeKey p l j.val →
      (r.dq,keyGradientAddress p l k j.val) ≠ (region,off)) → u.mem region off = initial.mem region off

theorem iteration (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hqo : r.q ≠ r.dq) (hko : r.k ≠ r.dq) (hvo : r.v ≠ r.dq) (hdo : r.do_ ≠ r.dq)
    (h : Holds p l initial r hasSeed t u) :
    ∃ v, stepStmts (BackwardSource.firstBody p)
      (u.setReg "i" .nat [] (Tile.scalar t)) = some v ∧
        Holds p l initial r hasSeed (t+1) v := by
  let entry := u.setReg "i" .nat [] (Tile.scalar t)
  let pre := beforeStore p l initial r hasSeed t entry
  let v := result p l initial r hasSeed t entry
  have he := BackwardFirstStep.executes p l initial r hasSeed t entry
    (ready_set_loop p l initial r hasSeed t t u h.ready)
    (by simpa [entry] using h.kmem) (by simpa [entry] using h.vmem)
    (by simpa [entry] using h.domem)
  refine ⟨v,he,?_,?_,?_,?_,?_,?_,?_⟩
  · exact ready_result p l initial r hasSeed t entry
      (ready_set_loop p l initial r hasSeed t t u h.ready)
  · intro off
    change (BackwardFirstState.scatter p l initial r hasSeed t pre).readMem r.q off = _
    rw [BackwardFirstState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hqo]
    simpa [pre,beforeStore,entry] using h.qmem off
  · intro off
    change (BackwardFirstState.scatter p l initial r hasSeed t pre).readMem r.k off = _
    rw [BackwardFirstState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hko]
    simpa [pre,beforeStore,entry] using h.kmem off
  · intro off
    change (BackwardFirstState.scatter p l initial r hasSeed t pre).readMem r.v off = _
    rw [BackwardFirstState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hvo]
    simpa [pre,beforeStore,entry] using h.vmem off
  · intro off
    change (BackwardFirstState.scatter p l initial r hasSeed t pre).readMem r.do_ off = _
    rw [BackwardFirstState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hdo]
    simpa [pre,beforeStore,entry] using h.domem off
  · intro k j hkt hj
    change (BackwardFirstState.scatter p l initial r hasSeed t pre).mem r.dq
      (keyGradientAddress p l k j.val) = _
    by_cases hkt' : k < t
    · rw [show (BackwardFirstState.scatter p l initial r hasSeed t pre).mem r.dq
        (keyGradientAddress p l k j.val) = pre.mem r.dq (keyGradientAddress p l k j.val) from
        Memory.full_frame p.BK r.dq (fun i => keyGradientAddress p l t i.val)
          (MaskedMath.dq p l initial r hasSeed t) (fun i => activeKey p l i.val)
          pre r.dq (keyGradientAddress p l k j.val) (by
            intro i hi heq
            have htk := (Addressing.key_gradient_unique p l t k i j hi hj
              (congrArg Prod.snd heq)).1
            omega)]
      simpa [pre,beforeStore,entry] using h.output k j hkt' hj
    · have hktEq : k = t := by omega
      subst k
      exact Memory.written_cell p.BK r.dq (fun i => keyGradientAddress p l t i.val)
        (MaskedMath.dq p l initial r hasSeed t) (fun i => activeKey p l i.val)
        pre j hj (fun i hi heq => (Addressing.key_gradient_unique p l t t i j hi hj heq).2)
  · intro region off hn
    change (BackwardFirstState.scatter p l initial r hasSeed t pre).mem region off = _
    rw [show (BackwardFirstState.scatter p l initial r hasSeed t pre).mem region off =
        pre.mem region off from
      Memory.full_frame p.BK r.dq (fun j => keyGradientAddress p l t j.val)
        (MaskedMath.dq p l initial r hasSeed t) (fun j => activeKey p l j.val)
        pre region off (fun j hj => hn t j (by omega) hj)]
    exact h.frame region off (fun k j hk hj => hn k j (by omega) hj)

theorem executes_remaining (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (hasSeed : Bool)
    (hqo : r.q ≠ r.dq) (hko : r.k ≠ r.dq) (hvo : r.v ≠ r.dq) (hdo : r.do_ ≠ r.dq)
    (remaining : Nat) : ∀ (t : Nat) (u : BlockState), t + remaining = p.T →
      Holds p l initial r hasSeed t u →
      ∃ v, stepForRangeAux "i" t p.T 1 (BackwardSource.firstBody p) u = some v ∧
        Holds p l initial r hasSeed p.T v := by
  induction remaining with
  | zero =>
      intro t u ht h
      have ht' : t = p.T := by omega
      subst t
      exact ⟨u,stepForRangeAux.step_ge (by decide) (by omega),h⟩
  | succ n ih =>
      intro t u ht h
      obtain ⟨v,hv,hr⟩ := iteration p l initial r hasSeed t u hqo hko hvo hdo h
      obtain ⟨w,hw,hwReady⟩ := ih (t+1) v (by omega) hr
      refine ⟨w,?_,hwReady⟩
      rw [stepForRangeAux.step_lt (by decide) (by omega),hv]
      exact hw

#axiomsClean iteration
#axiomsClean executes_remaining
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstLoop
