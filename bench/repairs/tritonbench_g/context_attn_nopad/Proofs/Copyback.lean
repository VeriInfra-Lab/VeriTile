import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPHostPrecise
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Copyback
open VeriTile.Triton FP.Structural
open FPHostSource
noncomputable section

/-- Ordinary in-place copy_ into the supplied logical view. Failure may leave
a partial copy: every destination coordinate is either unchanged or copied,
and every cell outside the destination view is unchanged. This is a tensor
coordinate contract on already computed values, not a kernel-result law. -/
structure API (α : Type) (c : Call) (view : OutputView α c) where
  copy : (Logical.Coord c → α) → (Nat → α) → Wrapper.StepResult (Nat → α)
  complete : ∀ values before after, copy values before = .ok after →
    ∀ i, view.action (after (view.address i)) = values i
  progress : ∀ values before i,
    (copy values before).state (view.address i) = before (view.address i) ∨
    view.action ((copy values before).state (view.address i)) = values i
  frame : ∀ values before addr, (¬ ∃ i, view.address i = addr) →
    (copy values before).state addr = before addr

theorem success_memory {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (view : OutputView α c) (api : API α c view) (raw : State α) (before after : Nat → α)
    (h : api.copy (fun i => (raw.mem r.out (Logical.coordOffset c i)).read .real) before = .ok after) :
    after = (finish r view before raw).memory := by
  funext addr
  dsimp only [finish,writeback]
  split
  next existsCoord =>
    let i := Classical.choose existsCoord
    have hi : view.address i = addr := Classical.choose_spec existsCoord
    have hc := api.complete _ before after h i
    have he := congrArg view.action hc
    rw [view.involution,hi] at he
    exact he
  next noCoord =>
    have hf := api.frame (fun i => (raw.mem r.out (Logical.coordOffset c i)).read .real) before addr noCoord
    simpa only [h,Wrapper.StepResult.state] using hf

/-- Initialized scratch makes even a failed partial copy preserve all cells
outside the exact logical kernel-write footprint, including inactive rows and
extra coordinates inside the destination view. -/
theorem outcome_frame {α : Type} {c : Call} (view : OutputView α c) (api : API α c view)
    (values : Logical.Coord c → α) (before : Nat → α)
    (stable : ∀ i, ¬ Logical.Written c i → values i = view.action (before (view.address i)))
    (addr : Nat) (unwritten : ¬ ∃ i, Logical.Written c i ∧ view.address i = addr) :
    (api.copy values before).state addr = before addr := by
  by_cases inside : ∃ i, view.address i = addr
  · obtain ⟨i,hi⟩ := inside
    have hnot : ¬ Logical.Written c i := fun hw => unwritten ⟨i,hw,hi⟩
    rcases api.progress values before i with unchanged | copied
    · simpa only [hi] using unchanged
    · rw [stable i hnot] at copied
      have he := view.involution.injective copied
      simpa only [hi] using he
  · exact api.frame values before addr inside

#print axioms success_memory
#print axioms outcome_frame
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Copyback
