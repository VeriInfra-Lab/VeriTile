import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.OutputNumerics

/-! Numerical-only finite/primitive obligations at the actual evaluated words.
These predicates do not constrain the total Word correctness theorems. They
range over all launched lanes, including inactive lanes that compute on the
source's zero-filled masked loads. Native scalar conformance and error-bound
validation remain explicitly deferred. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.OperandDomains
open Kernels ByteMemory Execution PrimitiveNumerics GraphNumerics TrajectoryNumerics OutputNumerics
set_option autoImplicit false
noncomputable section

def PairDomain (ops : Operations Word) (route : Route) (pair : Pair Expr) : Prop :=
  Domain ops pair.re ∧ (route = .complex64 → Domain ops pair.im)

def PairBounds (ops : Operations Word) (route : Route) (pair : Pair Expr) (bound : Atom → ℝ) : Prop :=
  (∀ atom ∈ pair.re.atoms ops, LocalBound ops atom (bound atom)) ∧
    (route = .complex64 → ∀ atom ∈ pair.im.atoms ops, LocalBound ops atom (bound atom))

def forward (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (m : Heap) : Prop :=
  ∀ pid, pid < NativeIndex.grid d.width → ∀ j : Lane, ∀ t, t < d.length →
    PairDomain ops route (forwardExpr route graph (forwardValues codec route graph ops d views pid m t j)
      (forwardTransition codec route ops d views pid m j) (forwardInput codec route ops d views pid m j t))

def backward (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (m : Heap) : Prop :=
  ∀ pid, pid < NativeIndex.grid d.width → ∀ j : Lane,
    (∀ i, i < d.length →
      let expr := backwardExpression codec route graph ops d views pid m j i
      PairDomain ops route expr.inputGradient ∧ PairDomain ops route expr.registers.carry ∧
        PairDomain ops route expr.registers.transition) ∧
    PairDomain ops route (terminalExpr route (backwardValues codec route graph ops d views pid m d.length j).carry) ∧
    PairDomain ops route (terminalExpr route (backwardValues codec route graph ops d views pid m d.length j).transition)

def reduction (ops : Operations Word) (batch dim components : Nat) (plan : Reduction.Plan batch dim components)
    (values : Nat → Nat → Word) : Prop :=
  ∀ q, q < components * dim → Domain ops (sumExpr (plan.tree q) (values q))

/-- Ordinary primitive conformance for two scalar backends. This quantifies
only scalar constants/operations at actual operands. It contains no execution,
shape, storage, launch-success or task-output equation. -/
def LocalConformance (observed model : Operations Word) (atoms : List Atom) : Prop :=
  ∀ atom ∈ atoms, atom.result observed = atom.result model

theorem expression_conformance (observed model : Operations Word) (expression : Expr)
    (agreement : LocalConformance observed model (expression.atoms observed)) :
    expression.evaluate observed = expression.evaluate model := by
  induction expression with
  | leaf word => rfl
  | zero => exact agreement .zero (by simp [Expr.atoms])
  | add a b ha hb =>
    have ea := ha (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have eb := hb (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have current := agreement (.add (a.evaluate observed) (b.evaluate observed)) (by simp [Expr.atoms])
    simpa [Expr.evaluate, Atom.result, ea, eb] using current
  | subtract a b ha hb =>
    have ea := ha (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have eb := hb (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have current := agreement (.subtract (a.evaluate observed) (b.evaluate observed)) (by simp [Expr.atoms])
    simpa [Expr.evaluate, Atom.result, ea, eb] using current
  | multiply a b ha hb =>
    have ea := ha (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have eb := hb (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have current := agreement (.multiply (a.evaluate observed) (b.evaluate observed)) (by simp [Expr.atoms])
    simpa [Expr.evaluate, Atom.result, ea, eb] using current
  | fma a b c ha hb hc =>
    have ea := ha (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have eb := hb (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have ec := hc (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have current := agreement (.fma (a.evaluate observed) (b.evaluate observed) (c.evaluate observed)) (by simp [Expr.atoms])
    simpa [Expr.evaluate, Atom.result, ea, eb, ec] using current
  | flipSign a ha =>
    have ea := ha (by intro atom h; exact agreement atom (by simp [Expr.atoms, h]))
    have current := agreement (.flipSign (a.evaluate observed)) (by simp [Expr.atoms])
    simpa [Expr.evaluate, Atom.result, ea] using current

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.OperandDomains
