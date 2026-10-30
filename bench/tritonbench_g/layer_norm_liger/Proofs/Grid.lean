import bench.tritonbench_g.layer_norm_liger.Proofs.ClosedFP
import bench.tritonbench_g.layer_norm_liger.Proofs.Frame
import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardContract
namespace VeriTile.Bench.Repairs.LayerNormLiger.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum softwareSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- A new program receives the current shared memory, and fresh thread-local
registers from the launch template. No previous program's locals are reused. -/
def launch {α : Type} (base : State α) (mem : RegionName → Nat → Cell α) (pid : Nat) : State α :=
  { base with mem := mem, pids := fun ax => if ax=0 then pid else base.pids ax }

def run {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel)
    (base : State α) : Nat → Option (State α)
  | 0 => some base
  | n+1 => do
      let s ← run M kernel base n
      let t ← FP.Structural.exec M kernel (launch base s.mem n)
      pure { base with mem := t.mem }

/-- Explicit sequential composition of complete per-program scalar plans. -/
def reference {α : Type} (step : State α → State α) (base : State α) : Nat → State α
  | 0 => base
  | n+1 => { base with mem := (step (launch base (reference step base n).mem n)).mem }

theorem complete {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel)
    (step : State α → State α)
    (one : ∀ s, ∃ t, FP.Structural.exec M kernel s=some t ∧ t.mem=(step s).mem)
    (base : State α) (programs : Nat) :
    run M kernel base programs=some (reference step base programs) := by
  induction programs with
  | zero => rfl
  | succ n ih =>
    obtain ⟨t,ht,hm⟩ := one (launch base (reference step base n).mem n)
    simp [run,ih,Option.bind,ht,reference,hm]

theorem frame {α : Type} (step : State α → State α) (base : State α)
    (programs : Nat) (r : RegionName) (o : Nat)
    (h : ∀ pid, pid<programs → ∀ mem,
      (step (launch base mem pid)).mem r o=mem r o) :
    (reference step base programs).mem r o=base.mem r o := by
  induction programs with
  | zero => rfl
  | succ n ih =>
    change (step (launch base (reference step base n).mem n)).mem r o=base.mem r o
    rw [h n (by omega)]
    exact ih (fun pid hp => h pid (by omega))

/-- A program's store survives all later programs when their footprints are
disjoint from that cell. This is a memory lemma, with no floating sum law. -/
theorem readback {α : Type} (step : State α → State α) (base : State α)
    (programs pid : Nat) (hp : pid<programs) (r : RegionName) (o : Nat)
    (h : ∀ later, pid<later → later<programs → ∀ mem,
      (step (launch base mem later)).mem r o=mem r o) :
    (reference step base programs).mem r o=
      (step (launch base (reference step base pid).mem pid)).mem r o := by
  induction programs with
  | zero => omega
  | succ n ih =>
    by_cases he : pid=n
    · subst pid; rfl
    · have hn : pid<n := by omega
      change (step (launch base (reference step base n).mem n)).mem r o=_
      rw [h n hn (by omega)]
      exact ih hn (fun later hl hu => h later hl (by omega))

theorem forward_complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Forward.Parameters) (base : State α) (rows : Nat) :
    run (withSum M R) (Forward.kernel p) base rows=
      some (reference (Forward.reference M R p) base rows) :=
  complete _ _ _ (Forward.complete_run M R p) base rows

theorem backward_complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Backward.Parameters) (base : State α) (programs : Nat) :
    run (withSum M R) (Backward.kernel p) base programs=
      some (reference (Backward.reference M R p) base programs) :=
  complete _ _ _ (Backward.complete_run M R p) base programs

theorem forward_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt : FP.Value .fp32 → FP.Value .fp32)
    (p : Forward.Parameters) (base : State (FP.Value .fp32)) (rows : Nat) :
    run (withSum (Forward.software rest cfg literal rsqrt) (softwareSum cfg))
      (Forward.kernel p) base rows=
      some (reference (ClosedFP.forwardReference cfg literal rsqrt p) base rows) :=
  complete _ _ _ (ClosedFP.forward_exec rest cfg literal rsqrt p) base rows

theorem backward_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt : FP.Value .fp32 → FP.Value .fp32)
    (p : Backward.Parameters) (base : State (FP.Value .fp32)) (programs : Nat) :
    run (withSum (Forward.software rest cfg literal rsqrt) (softwareSum cfg))
      (Backward.kernel p) base programs=
      some (reference (ClosedFP.backwardReference cfg literal p) base programs) :=
  complete _ _ _ (ClosedFP.backward_exec rest cfg literal rsqrt p) base programs

#axiomsClean complete
#axiomsClean frame
#axiomsClean readback
#axiomsClean forward_complete
#axiomsClean backward_complete
#axiomsClean forward_bits
#axiomsClean backward_bits
end
end VeriTile.Bench.Repairs.LayerNormLiger.Grid
