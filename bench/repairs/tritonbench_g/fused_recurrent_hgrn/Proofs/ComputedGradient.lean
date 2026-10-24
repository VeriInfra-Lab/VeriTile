import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Gradient
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000

def realEntry (memory : RegionName → Nat → ℝ) : State ℝ where
  mem := fun r a => .mk .real (memory r a)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def realInput (memory : RegionName → Nat → ℝ) (T D : Nat) : Input ℝ := fun r bh t d =>
  memory r (if r="h0" ∨ r="dht" then stateAddr bh D d else rowAddr bh T D t d)

theorem real_entry_inputs (memory : RegionName → Nat → ℝ) (T D : Nat) :
    Execution.Inputs (realEntry memory) (realInput memory T D) T D := by
  constructor
  · intro r hr bh t d
    rcases hr with rfl|rfl|rfl <;> simp [Execution.read,realEntry,realInput,Cell.read]
  · intro r hr bh d
    rcases hr with rfl|rfl <;> simp [Execution.read,realEntry,realInput,Cell.read]

/-- Actual source buffers supply the derivative of the independently defined
loss, for every direction and every active lane. h0 is held constant by pathState. -/
theorem real_host_and_gradient (memory : RegionName → Nat → ℝ) (upstream finalUpstream : Nat → ℝ)
    (BH T D BDF BDB : Nat) (hF : 0 < BDF) (hB : 0 < BDB) (U S hasDo hasDht : Bool) :
    let input := realInput memory T D
    let copied := Execution.hostInput realOps input T D hasDo upstream finalUpstream
    ∃ f b,
      Execution.gridRun realOps (forward T D BDF U S).surfaceBody
        (Execution.programs BH D BDF) (realEntry memory)=some f ∧
      Execution.gridRun realOps (backward T D BDB U (S && hasDht)).surfaceBody
        (Execution.programs BH D BDB) (Execution.installCotangents realOps f hasDo upstream finalUpstream)=some b ∧
      Execution.Saved realOps b input U BH T D ∧
      Execution.Gradients realOps b copied U (S && hasDht) BH T D ∧
      (S=Bool.true → ∀ bh, bh < BH → ∀ d, d < D →
        b.mem "ht" (stateAddr bh D d)=.mk .real (state realOps U input bh d T)) ∧
      (∀ bh, bh < BH → ∀ d, d < D → ∀ vx vg : Nat → ℝ,
        HasDerivAt (fun e =>
          (∑ t ∈ Finset.range T, copied "do" bh t d*pathState copied U bh d vx vg e (t+1))+
            terminal realOps (S && hasDht) copied bh d*pathState copied U bh d vx vg e T)
          (∑ t ∈ Finset.range T,
            (Execution.read b "dx" (rowAddr bh T D t d)*vx t+
             Execution.read b "dg" (rowAddr bh T D t d)*vg t)) 0) ∧
      (∀ r a, r≠("do" : RegionName) → r≠("dht" : RegionName) →
        ¬ Execution.ForwardRegion BH T D S r a → ¬ Execution.BackwardRegion BH T D r a →
        b.mem r a=(realEntry memory).mem r a) := by
  dsimp only
  obtain ⟨f,b,hf,hb,hO,hG,hH,hframe⟩ := Execution.host_result realOps (realInput memory T D)
    BH T D BDF BDB hF hB U S hasDo hasDht upstream finalUpstream (realEntry memory) (real_entry_inputs memory T D)
  refine ⟨f,b,hf,hb,hO,hG,hH,?_,hframe⟩
  intro bh hbh d hd vx vg
  have gradients :
      (∑ t ∈ Finset.range T, (Execution.read b "dx" (rowAddr bh T D t d)*vx t+
        Execution.read b "dg" (rowAddr bh T D t d)*vg t))=
      (∑ t ∈ Finset.range T, (dx realOps (S && hasDht)
        (Execution.hostInput realOps (realInput memory T D) T D hasDo upstream finalUpstream) T bh d t*vx t+
        dg realOps U (S && hasDht)
        (Execution.hostInput realOps (realInput memory T D) T D hasDo upstream finalUpstream) T bh d t*vg t)) := by
    apply Finset.sum_congr rfl
    intro t ht
    obtain ⟨hx,hg⟩ := hG bh hbh t (Finset.mem_range.mp ht) d hd
    unfold Execution.read
    rw [hx,hg]
    rfl
  rw [gradients]
  exact loss_derivative _ U (S && hasDht) T bh d vx vg

#axiomsClean real_host_and_gradient
#specNonCircular realInput avoiding [FP.Structural.exec,Execution.gridRun]
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
