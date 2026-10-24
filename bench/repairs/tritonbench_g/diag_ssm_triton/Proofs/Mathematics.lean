import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Adjoint

/-! Independent public array mathematics. Lambda is shared over batch, so its
cotangent is a batch sum. Real32 is the real-valued slice of this complex
real-inner-product adjoint, not a different complex-sign convention. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Mathematics
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

def state (s : Nat → Nat → ℂ) (transition : Nat → ℂ) (x : Nat → Nat → Nat → ℂ)
    (batch dim time : Nat) : ℂ := Reference.forwardState (s batch dim) (transition dim) (fun t => x t batch dim) time

def output (s : Nat → Nat → ℂ) (transition : Nat → ℂ) (x : Nat → Nat → Nat → ℂ)
    (time batch dim : Nat) : ℂ := state s transition x batch dim (time + 1)

def initialGradient (length : Nat) (transition : Nat → ℂ) (gy : Nat → Nat → Nat → ℂ)
    (batch dim : Nat) : ℂ := Adjoint.carry length (transition dim) (fun t => gy t batch dim) length

def inputGradient (length : Nat) (transition : Nat → ℂ) (gy : Nat → Nat → Nat → ℂ)
    (time batch dim : Nat) : ℂ := Reference.inputGradient length (starRingEnd ℂ (transition dim)) (fun t => gy t batch dim) time

def transitionContribution (length : Nat) (s : Nat → Nat → ℂ) (transition : Nat → ℂ)
    (x gy : Nat → Nat → Nat → ℂ) (batch dim : Nat) : ℂ :=
  Adjoint.transitionGradient length (transition dim) (state s transition x batch dim) (fun t => gy t batch dim) length

def transitionGradient (length batch : Nat) (s : Nat → Nat → ℂ) (transition : Nat → ℂ)
    (x gy : Nat → Nat → Nat → ℂ) (dim : Nat) : ℂ :=
  (Finset.range batch).sum (fun b => transitionContribution length s transition x gy b dim)

theorem closed_form (s : Nat → Nat → ℂ) (transition : Nat → ℂ) (x : Nat → Nat → Nat → ℂ)
    (time batch dim : Nat) :
    output s transition x time batch dim = s batch dim * transition dim ^ (time + 1) +
      ∑ k ∈ Finset.range (time + 1), x k batch dim * transition dim ^ (time - k) := by
  simpa [output, state] using Reference.forward_closed_form (s batch dim) (transition dim) (fun t => x t batch dim) (time + 1)

/-- Independent first-order adjoint for all batches at one dimension. A sum of
this identity over dim gives the full tensor pairing. The transition term is
exactly the returned batch reduction, including batch=0. -/
theorem batch_adjoint (length batch dim : Nat) (s ds : Nat → Nat → ℂ)
    (transition dtransition : Nat → ℂ) (x gy dx : Nat → Nat → Nat → ℂ) :
    (Finset.range batch).sum (fun b => Adjoint.outputPairings length (fun t => gy t b dim)
      (Adjoint.tangent (ds b dim) (dtransition dim) (transition dim) (state s transition x b dim)
        (fun t => dx t b dim)) length) =
    (Finset.range batch).sum (fun b => Adjoint.pairing (initialGradient length transition gy b dim) (ds b dim) +
      Adjoint.inputPairings length (transition dim) (fun t => gy t b dim) (fun t => dx t b dim) length) +
    Adjoint.pairing (transitionGradient length batch s transition x gy dim) (dtransition dim) := by
  simp only [initialGradient, transitionGradient, transitionContribution,
    ← Adjoint.batch_adjoint batch _ (dtransition dim), ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro b hb
  exact Adjoint.whole_adjoint length (transition dim) (dtransition dim) (ds b dim)
    (state s transition x b dim) (fun t => gy t b dim) (fun t => dx t b dim)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.Mathematics
