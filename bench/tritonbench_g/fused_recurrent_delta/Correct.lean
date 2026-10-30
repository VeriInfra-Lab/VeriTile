import bench.tritonbench_g.fused_recurrent_delta.Proofs.LogicalCoordinates
import VeriTile.Meta.Specification

/-! Each complete original program implements the independent delta recurrence
and its value-block VJP contributions at the explicit `(bh, iv)` program ID.
DQ, DK and scalar DBeta are this program's partial sums over its BBV value lanes;
no cross-program host reduction is claimed. IO.lean exposes all input/output
windows and the in-place residual. MathematicalIO.lean supplies the canonical
formulas and saved-input relation reused by the displayed mathematics below. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
open VeriTile.Triton FP.Structural Interface
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Independent delta step, for each value column: r[t] = x[t] - Σᵢ h[t,i]k[t,i],
h[t+1,i] = h[t,i] + k[t,i]β[t]r[t]. Initial state and scalar/vector beta
are selected by `primal`; the public outputs are O, residual V, and optional Ht. -/
def forwardValues (s : Shape) (f : Flags) (scale : ℝ) (vals : ForwardTensorInputs ℝ s) : ForwardTensorOutputs ℝ s where
  output := fun t v =>
    let a := forwardPrimal s f vals v
    scale * ∑ i, Math.state a (t.val+1) i * a.q t.val i
  residual := fun t v =>
    let a := forwardPrimal s f vals v
    a.x t.val - ∑ i, Math.state a t.val i * a.k t.val i
  finalState := fun k v => if hi : k.val<s.K then
    Math.state (forwardPrimal s f vals v) s.T ⟨k.val,hi⟩ else 0

/-- Pure VJP at time t. The reverse carry is seeded by `terminal` at T;
E = carry[t+1] + scale*g[t]*q[t] and z = Σᵢ E[i]k[t,i]. -/
def deltaGradient {K : Nat} (scale : ℝ) (a : Math.Input K) (g : Nat → ℝ)
    (terminal : Math.Vec K) (T t : Nat) : Math.Grad K :=
  let h := Math.state a t
  let r := a.x t - ∑ i, h i * a.k t i
  let E := fun i => Math.carry scale a g terminal T (t+1) i + scale*g t*a.q t i
  let z := ∑ i, E i * a.k t i
  { q := fun i => scale*g t*(h i+a.k t i*(a.beta t*r))
    k := fun i => (a.beta t*r)*E i-(a.beta t*z)*h i
    x := a.beta t*z
    beta := r*z
    h := fun i => E i-(a.beta t*z)*a.k t i }

/-- All six backward channels: DQ, DK, DV, vector DBeta, scalar DBeta and DH0.
Only used-time saved residuals are constrained, by `SavedInputs` below. -/
def backwardValues (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (_ : BackwardTensorInputs ℝ s) : BackwardTensorOutputs ℝ s where
  dq := fun t k => ∑ v : Fin s.BBV, (deltaGradient scale (a v) (fun t => g t v) (terminal v) s.T t.val).q k
  dk := fun t k => ∑ v : Fin s.BBV, (deltaGradient scale (a v) (fun t => g t v) (terminal v) s.T t.val).k k
  dv := fun t v => (deltaGradient scale (a v) (fun t => g t v) (terminal v) s.T t.val).x
  vectorDBeta := fun t v => (deltaGradient scale (a v) (fun t => g t v) (terminal v) s.T t.val).beta
  scalarDBeta := fun t => ∑ v : Fin s.BBV, (deltaGradient scale (a v) (fun t => g t v) (terminal v) s.T t.val).beta
  dh0 := fun k v => Math.carry scale (a v) (fun t => g t v) (terminal v) s.T 0 k

specification fused_recurrent_delta_forward_correctness (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) :
    Spec.Real (forwardLogicalIO s f scale bh iv ⊨
      forwardValues s f scale) :=
  LogicalProof.forward s f scale bh iv

specification fused_recurrent_delta_backward_correctness (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) :
    Spec.Real (backwardLogicalIO s f scale bh iv ⊨{savedDomain s f iv a g terminal}
      backwardValues s scale a g terminal) :=
  LogicalProof.backward s f scale bh iv a g terminal

#stmtSurfaceSubset fused_recurrent_delta_forward_correctness ⊆
  [Spec.Real,Shape,Flags,forwardLogicalIO,ForwardTensorInputs,ForwardTensorOutputs,LogicalProgramIO.Implements,forwardValues]
#stmtSurfaceSubset fused_recurrent_delta_backward_correctness ⊆
  [Spec.Real,Shape,Shape.BBV,Shape.BK,Flags,Math.Input,Math.Vec,
    backwardLogicalIO,BackwardTensorInputs,BackwardTensorOutputs,LogicalProgramIO.ImplementsOn,savedDomain,backwardValues]
#axiomsClean fused_recurrent_delta_forward_correctness
#axiomsClean fused_recurrent_delta_backward_correctness
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
