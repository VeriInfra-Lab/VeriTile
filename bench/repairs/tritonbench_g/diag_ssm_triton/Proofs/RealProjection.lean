import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.RealGraph

/-! Independent Real interpretation of the complete scalar scans. This uses
abstract mathematical values, never an impossible finite-byte codec for ℝ.
The Word wrapper theorems establish actual placement, returned storage and
frames independently. The unchanged complete typed Real IO proofs are reused
in PerProgram; here their scalar target is identified with the adjoint. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.RealProjection
open Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

def embed (route : Route) (pair : Pair ℝ) : ℂ :=
  match route with
  | .real32 => ⟨pair.re, 0⟩
  | .complex64 => ⟨pair.re, pair.im⟩

def observe (route : Route) (pair : Pair ℝ) : ℂ :=
  match route with
  | .real32 => embed route pair
  | .complex64 => starRingEnd ℂ (embed route pair)

def scanForward (route : Route) (graph : Graph) (initial transition : Pair ℝ) (input : Nat → Pair ℝ) : Nat → Pair ℝ
  | 0 => initial
  | t + 1 => Kernels.forward route graph RealGraph.arithmetic (scanForward route graph initial transition input t) transition (input t)

theorem forward_step (route : Route) (graph : Graph) (s transition x : Pair ℝ) :
    embed route (Kernels.forward route graph RealGraph.arithmetic s transition x) =
      embed route s * embed route transition + embed route x := by
  cases route <;> cases graph <;> apply Complex.ext <;>
    simp [embed, Kernels.forward, realForward, complexForward, complexProduct,
      RealGraph.arithmetic, Complex.mul_re, Complex.mul_im] <;> ring

theorem forward (route : Route) (graph : Graph) (initial transition : Pair ℝ) (input : Nat → Pair ℝ) (t : Nat) :
    embed route (scanForward route graph initial transition input t) =
      Reference.forwardState (embed route initial) (embed route transition) (fun q => embed route (input q)) t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [scanForward, forward_step, ih, Reference.forwardState]

theorem forward_closed (route : Route) (graph : Graph) (initial transition : Pair ℝ) (input : Nat → Pair ℝ) (t : Nat) :
    embed route (scanForward route graph initial transition input t) =
      embed route initial * embed route transition ^ t +
      ∑ k ∈ Finset.range t, embed route (input k) * embed route transition ^ (t - 1 - k) := by
  rw [forward, Reference.forward_closed_form]

def scanBackward (route : Route) (graph : Graph) (length : Nat) (transition : Pair ℝ)
    (previous gy : Nat → Pair ℝ) : Nat → BackwardRegisters ℝ
  | 0 => initialBackward RealGraph.arithmetic
  | i + 1 => (Kernels.backward route graph RealGraph.arithmetic (previous (length - 1 - i)) transition
      (gy (length - 1 - i)) (scanBackward route graph length transition previous gy i)).registers

def scanInputGradient (route : Route) (graph : Graph) (length : Nat) (transition : Pair ℝ)
    (previous gy : Nat → Pair ℝ) (i : Nat) : Pair ℝ :=
  (Kernels.backward route graph RealGraph.arithmetic (previous (length - 1 - i)) transition
    (gy (length - 1 - i)) (scanBackward route graph length transition previous gy i)).inputGradient

theorem backward_step (route : Route) (graph : Graph) (previous transition gy : Pair ℝ) (r : BackwardRegisters ℝ) :
    let next := Kernels.backward route graph RealGraph.arithmetic previous transition gy r
    embed route next.inputGradient = embed route gy + observe route r.carry ∧
      observe route next.registers.carry = (embed route gy + observe route r.carry) * starRingEnd ℂ (embed route transition) ∧
      observe route next.registers.transition = observe route r.transition +
        (embed route gy + observe route r.carry) * starRingEnd ℂ (embed route previous) := by
  cases route <;> cases graph
  all_goals dsimp only
  all_goals constructor
  all_goals try constructor
  all_goals apply Complex.ext <;>
    simp [Kernels.backward, realBackward, complexBackward, accumulatedGradient, complexProduct,
      gradientConjugate, RealGraph.arithmetic, embed, observe, Complex.mul_re, Complex.mul_im] <;> try ring
  all_goals simp

theorem backward (route : Route) (graph : Graph) (length : Nat) (transition : Pair ℝ)
    (previous gy : Nat → Pair ℝ) (i : Nat) :
    observe route (scanBackward route graph length transition previous gy i).carry =
      Adjoint.carry length (embed route transition) (fun t => embed route (gy t)) i ∧
    observe route (scanBackward route graph length transition previous gy i).transition =
      Adjoint.transitionGradient length (embed route transition) (fun t => embed route (previous t))
        (fun t => embed route (gy t)) i := by
  induction i with
  | zero =>
    cases route <;> simp [scanBackward, observe, embed, initialBackward, zeroPair,
      RealGraph.arithmetic, Adjoint.carry, Reference.reverseCarry, Adjoint.transitionGradient] <;> rfl
  | succ i ih =>
    have step := backward_step route graph (previous (length - 1 - i)) transition (gy (length - 1 - i))
      (scanBackward route graph length transition previous gy i)
    constructor
    · change observe route (Kernels.backward route graph RealGraph.arithmetic _ _ _ _).registers.carry = _
      rw [step.2.1, ih.1]
      rfl
    · change observe route (Kernels.backward route graph RealGraph.arithmetic _ _ _ _).registers.transition = _
      rw [step.2.2, ih.1, ih.2]
      rfl

theorem input_gradient (route : Route) (graph : Graph) (length : Nat) (transition : Pair ℝ)
    (previous gy : Nat → Pair ℝ) (i : Nat) :
    embed route (scanInputGradient route graph length transition previous gy i) =
      Adjoint.inputGradient length (embed route transition) (fun t => embed route (gy t)) i := by
  unfold scanInputGradient
  rw [(backward_step route graph _ _ _ _).1, (backward route graph length transition previous gy i).1]
  rfl

theorem terminal (route : Route) (pair : Pair ℝ) :
    embed route (terminalGradient route RealGraph.arithmetic pair) = observe route pair := by
  cases route <;> apply Complex.ext <;> simp [embed, observe, terminalGradient, gradientConjugate, RealGraph.arithmetic]

/-- Complete time scan, including length zero, with the source terminal
conjugation applied exactly once to each returned scalar gradient. -/
theorem returned_gradients (route : Route) (graph : Graph) (length : Nat) (transition : Pair ℝ)
    (previous gy : Nat → Pair ℝ) :
    embed route (terminalGradient route RealGraph.arithmetic
      (scanBackward route graph length transition previous gy length).carry) =
        Adjoint.carry length (embed route transition) (fun t => embed route (gy t)) length ∧
    embed route (terminalGradient route RealGraph.arithmetic
      (scanBackward route graph length transition previous gy length).transition) =
        Adjoint.transitionGradient length (embed route transition) (fun t => embed route (previous t))
          (fun t => embed route (gy t)) length := by
  simp only [terminal]
  exact backward route graph length transition previous gy length

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.RealProjection
