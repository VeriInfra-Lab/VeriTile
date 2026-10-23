import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Kernel
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectHost

namespace VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
set_option maxHeartbeats 4000000

abbrev ReturnedTensor := SwigluTriton.DirectHost.ReturnedTensor
abbrev returned (p : Prepared backward) := SwigluTriton.DirectHost.returned p
abbrev Outcome := SwigluTriton.DirectHost.Outcome

export Kernel (Plan Outside Values RunSpec run_spec)

structure Runtime (backward : Bool) where
  prepare : Except String (Prepared backward)
  setup : (p : Prepared backward) → Except String (Plan p)

def execute {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : Channel → α → α → α → α) (rt : Runtime backward) (s : M) : Outcome M :=
  match rt.prepare with
  | .error reason => .failure reason
  | .ok p =>
    if hc : inputCheck p = Bool.true then
      match rt.setup p with
      | .error reason => .failure reason
      | .ok plan => .returned (returned p)
        (Physical.run ops compute backward (layout p ((inputCheck_admitted_iff p).mp hc)) plan.order s)
    else .failure "input contract"

/-- Complete returned values, exterior bytes and the source certificates are
coupled to the same physical state sequence. No tile-result hypothesis occurs. -/
def HostSpec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (formula : Channel → α → α → α → α)
    (linked : (p : Prepared backward) → Admitted p → Plan p → M → Prop)
    (rt : Runtime backward) (entry : M) (outcome : Outcome M) : Prop :=
  (∃ reason, rt.prepare = .error reason ∧ outcome = .failure reason) ∨
  (∃ p, rt.prepare = .ok p ∧ inputCheck p = Bool.false ∧ outcome = .failure "input contract") ∨
  (∃ p reason, rt.prepare = .ok p ∧ inputCheck p = Bool.true ∧
    rt.setup p = .error reason ∧ outcome = .failure reason) ∨
  (∃ p h plan final, rt.prepare = .ok p ∧ rt.setup p = .ok plan ∧
    outcome = .returned (returned p) final ∧ RunSpec ops formula p h entry final ∧ linked p h plan entry ∧
    Grid.grid (p.rows*p.cols) < 2^31 ∧
    (∀ pid lane, pid < Grid.grid (p.rows*p.cols) → lane < 128 →
      (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).toInt = (pid*128+lane : Nat) ∧
      (BitVec.ofNat 64 pid * BitVec.ofNat 64 128 + BitVec.ofNat 64 lane).slt
        (BitVec.ofNat 64 (p.rows*p.cols)) = decide (pid*128+lane < p.rows*p.cols)) ∧
    (∀ r i, i < p.rows*p.cols →
      (BitVec.ofNat 64 (p.tensor r).metadata.dataPtr.toNat + BitVec.ofNat 64 (4*i)).toNat =
        ((p.tensor r).key i).address ∧ ((p.tensor r).key i).address % 4 = 0))

theorem host_spec {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute formula : Channel → α → α → α → α) (same : ∀ ch a b d, compute ch a b d = formula ch a b d)
    (linked : (p : Prepared backward) → Admitted p → Plan p → M → Prop)
    (source : ∀ p h plan s, linked p h plan s) (rt : Runtime backward) (s : M) :
    HostSpec ops formula linked rt s (execute ops compute rt s) := by
  cases hp : rt.prepare with
  | error reason => exact Or.inl ⟨reason,hp,by simp [execute,hp]⟩
  | ok p =>
    cases hc : inputCheck p with
    | «false» => exact Or.inr (Or.inl ⟨p,hp,hc,by simp [execute,hp,hc]⟩)
    | «true» =>
      cases setup : rt.setup p with
      | error reason => exact Or.inr (Or.inr (Or.inl ⟨p,reason,hp,hc,setup,by simp [execute,hp,hc,setup]⟩))
      | ok plan =>
        let h := (inputCheck_admitted_iff p).mp hc
        apply Or.inr; apply Or.inr; apply Or.inr
        refine ⟨p,h,plan,_,hp,setup,?_,run_spec ops compute formula same p h plan s,
          source p h plan s,h.grid32,indices p h,pointer p h⟩
        simp [execute,hp,hc,setup,h]

#axiomsClean host_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.DirectHost
