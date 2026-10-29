import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference

/-! Independent mathematics. The reverse recurrence is the real-inner-product
adjoint of the first-order perturbation of the forward recurrence. No floating
kernel, memory, compiler or successful-execution law is an input here. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Adjoint
noncomputable section
open scoped BigOperators
open VeriTile.Bench.TritonBenchG.DiagSsmTriton

def pairing (u v : ℂ) : ℝ := (starRingEnd ℂ u * v).re

@[simp] theorem pairing_add_left (u v w : ℂ) :
    pairing (u + v) w = pairing u w + pairing v w := by
  simp [pairing, add_mul]

@[simp] theorem pairing_add_right (u v w : ℂ) :
    pairing u (v + w) = pairing u v + pairing u w := by
  simp [pairing, mul_add]

@[simp] theorem pairing_zero_left (u : ℂ) : pairing 0 u = 0 := by simp [pairing]

theorem step_adjoint (g dh h transition dtransition dx : ℂ) :
    pairing g (dh * transition + h * dtransition + dx) =
      pairing (g * starRingEnd ℂ transition) dh +
      pairing (g * starRingEnd ℂ h) dtransition + pairing g dx := by
  simp only [pairing, map_mul, starRingEnd_apply, star_star]
  simp only [Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im]
  ring

/-- Local first-order coefficient, with the exact quadratic remainder retained. -/
theorem step_perturbation (h dh transition dtransition x dx ε : ℂ) :
    (h + ε * dh) * (transition + ε * dtransition) + (x + ε * dx) =
      (h * transition + x) + ε * (dh * transition + h * dtransition + dx) +
      ε ^ 2 * (dh * dtransition) := by ring

def tangent (initial dtransition transition : ℂ) (state input : Nat → ℂ) : Nat → ℂ
  | 0 => initial
  | t + 1 => tangent initial dtransition transition state input t * transition +
      state t * dtransition + input t

def carry (length : Nat) (transition : ℂ) (gy : Nat → ℂ) : Nat → ℂ :=
  Reference.reverseCarry length (starRingEnd ℂ transition) gy

def inputGradient (length : Nat) (transition : ℂ) (gy : Nat → ℂ) (k : Nat) : ℂ :=
  gy (length - 1 - k) + carry length transition gy k

def transitionGradient (length : Nat) (transition : ℂ) (state gy : Nat → ℂ) : Nat → ℂ
  | 0 => 0
  | k + 1 => transitionGradient length transition state gy k +
      inputGradient length transition gy k * starRingEnd ℂ (state (length - 1 - k))

def outputPairings (length : Nat) (gy perturbation : Nat → ℂ) (k : Nat) : ℝ :=
  ∑ i ∈ Finset.range k, pairing (gy (length - 1 - i)) (perturbation (length - i))

def inputPairings (length : Nat) (transition : ℂ) (gy dx : Nat → ℂ) (k : Nat) : ℝ :=
  ∑ i ∈ Finset.range k, pairing (inputGradient length transition gy i) (dx (length - 1 - i))

/-- Reverse-prefix telescoping; the boundary carry is explicit. -/
theorem prefix_adjoint (length : Nat) (transition dtransition ds : ℂ)
    (state gy dx : Nat → ℂ) (k : Nat) (hk : k ≤ length) :
    let dh := tangent ds dtransition transition state dx
    outputPairings length gy dh k =
      pairing (carry length transition gy k) (dh (length - k)) +
      inputPairings length transition gy dx k +
      pairing (transitionGradient length transition state gy k) dtransition := by
  dsimp only
  induction k with
  | zero => simp [outputPairings, inputPairings, carry, Reference.reverseCarry,
      transitionGradient]
  | succ k ih =>
    have hkl : k ≤ length := by omega
    have ht : length - k = (length - 1 - k) + 1 := by omega
    have hn : length - (k + 1) = length - 1 - k := by omega
    have ih := ih hkl
    rw [outputPairings, Finset.sum_range_succ]
    change outputPairings length gy (tangent ds dtransition transition state dx) k +
      pairing (gy (length - 1 - k))
        (tangent ds dtransition transition state dx (length - k)) = _
    rw [ih, ht, tangent]
    have hstep := step_adjoint (inputGradient length transition gy k)
      (tangent ds dtransition transition state dx (length - 1 - k))
      (state (length - 1 - k)) transition dtransition (dx (length - 1 - k))
    simp only [inputGradient, pairing_add_left] at hstep
    simp only [inputPairings, Finset.sum_range_succ, transitionGradient,
      pairing_add_left, hn]
    change _ = pairing ((gy (length - 1 - k) + carry length transition gy k) *
        starRingEnd ℂ transition)
        (tangent ds dtransition transition state dx (length - 1 - k)) + _ + _
    simp only [inputGradient, pairing_add_left]
    linarith [hstep]

/-- The whole reverse scan is the adjoint of the independent first-order
forward perturbation, for arbitrary complex cotangents and all lengths. -/
theorem whole_adjoint (length : Nat) (transition dtransition ds : ℂ)
    (state gy dx : Nat → ℂ) :
    outputPairings length gy (tangent ds dtransition transition state dx) length =
      pairing (carry length transition gy length) ds +
      inputPairings length transition gy dx length +
      pairing (transitionGradient length transition state gy length) dtransition := by
  simpa [tangent] using prefix_adjoint length transition dtransition ds state gy dx length le_rfl

/-- The same time-indexed independent gradient already used by the original
per-program execution theorem. -/
theorem input_gradient_reference (length : Nat) (transition : ℂ)
    (gy : Nat → ℂ) (t : Nat) (ht : t < length) :
    inputGradient length transition gy (length - 1 - t) =
      Reference.inputGradient length (starRingEnd ℂ transition) gy t := by
  have h : length - 1 - (length - 1 - t) = t := by omega
  simp [inputGradient, Reference.inputGradient, carry, h]

theorem previous_state (initial transition : ℂ) (input : Nat → ℂ) (t : Nat) :
    Reference.previousState (starRingEnd ℂ initial)
      (fun q => starRingEnd ℂ (Reference.forwardState initial transition input (q + 1))) t =
      starRingEnd ℂ (Reference.forwardState initial transition input t) := by
  cases t with
  | zero => simp [Reference.previousState, Reference.forwardState]
  | succ t => simp [Reference.previousState]

theorem transition_gradient_reference (length : Nat) (initial transition : ℂ)
    (input gy : Nat → ℂ) (k : Nat) :
    transitionGradient length transition (Reference.forwardState initial transition input) gy k =
      Reference.transitionGradient length (starRingEnd ℂ transition) (starRingEnd ℂ initial)
        (fun q => starRingEnd ℂ (Reference.forwardState initial transition input (q + 1))) gy k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [transitionGradient, Reference.transitionGradient, ih, previous_state,
      inputGradient, carry]

theorem pairing_sum_left (n : Nat) (g : Nat → ℂ) (delta : ℂ) :
    pairing (∑ b ∈ Finset.range n, g b) delta =
      ∑ b ∈ Finset.range n, pairing (g b) delta := by
  induction n with
  | zero => simp
  | succ n ih => simp [Finset.sum_range_succ, ih]

/-- A shared transition perturbation collects over batch into exactly the
dimension-zero sum, including the empty batch. -/
theorem batch_adjoint (batch : Nat) (g : Nat → ℂ) (delta : ℂ) :
    (∑ b ∈ Finset.range batch, pairing (g b) delta) =
      pairing (∑ b ∈ Finset.range batch, g b) delta :=
  (pairing_sum_left batch g delta).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.Adjoint
