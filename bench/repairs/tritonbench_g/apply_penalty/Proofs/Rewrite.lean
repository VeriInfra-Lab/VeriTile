import bench.repairs.tritonbench_g.apply_penalty.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Value
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ApplyPenalty.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
open FP.ProfiledRewrite (engine)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def originalSource (stride B : Nat) : ComputeKernel :=
  kernel "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum" stride 1 B

def reciprocalSource (stride B : Nat) : ComputeKernel :=
  reciprocalKernel "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum" stride 1 B

/-- Comparison availability is a semantic backend requirement, not a numerical
rewrite assumption and not a Python guard. -/
def Supported {α : Type} (M : Algebra α) : Prop :=
  ∃ lt, M.compareLt (some .fp32) .real = some lt

def originalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (lt : α → α → Bool) (hlt : M.compareLt (some .fp32) .real = some lt)
    (stride B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (originalSource stride B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [originalSource, kernel, FP.Structural.exec, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ComputeDType.eraseDType, Option.bind, numeric, numericLt, natLt, bop,
    store, Region.cast, engine, Algebra.withDefaultPrecision, resolvePrecision, hlt]
  rfl

def reciprocalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (lt : α → α → Bool) (hlt : M.compareLt (some .fp32) .real = some lt)
    (stride B : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (reciprocalSource stride B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [reciprocalSource, reciprocalKernel, FP.Structural.exec, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ComputeDType.eraseDType, Option.bind, numeric, numericLt, natLt, bop,
    store, Region.cast, engine, Algebra.withDefaultPrecision, resolvePrecision, hlt]
  rfl


def start {α : Type} [Inhabited α] (s : State α) : Nat :=
  (s.mem "cumsum" (s.pids 0)).read .nat

def stop {α : Type} [Inhabited α] (s : State α) : Nat :=
  (s.mem "cumsum" (s.pids 0 + 1)).read .nat

def active {α : Type} [Inhabited α] (s : State α) (i : TileIndex [B]) : Prop :=
  start s + i.1.val < stop s

instance {α : Type} [Inhabited α] (s : State α) (i : TileIndex [B]) :
    Decidable (active s i) := inferInstanceAs (Decidable (start s + i.1.val < stop s))

def address {α : Type} [Inhabited α] (stride : Nat) (s : State α) (i : TileIndex [B]) : Nat :=
  s.pids 0 * stride + if active s i then (s.mem "ids" (start s + i.1.val)).read .nat else 0

def loadedLogit {α : Type} [Inhabited α] (M : Algebra α)
    (stride : Nat) (s : State α) (i : TileIndex [B]) : α :=
  M.fp32Load (if active s i then (s.mem "logits" (address stride s i)).read .real
    else M.literal (some .fp32) .real 0.0)

def loadedRepetition {α : Type} [Inhabited α] (M : Algebra α) (s : State α) : α :=
  M.fp32Load ((s.mem "repetition" (s.pids 0)).read .real)

/-- Exactly the operands whose rewritten positive arm reaches an active store.
Inactive lanes and nonpositive/NaN comparisons impose no numerical condition.
This is a transformation domain, independent of the Python input contract. -/
def DivisionDomain {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (lt : α → α → Bool) (stride B : Nat) (s : State α) : Prop :=
  ∀ i : TileIndex [B], active s i →
    lt (M.literal (some .fp32) .real 0) (loadedLogit M stride s i) = Bool.true →
      D .finite (loadedLogit M stride s i) ∧
      D .finite (loadedRepetition M s) ∧ D .nonzero (loadedRepetition M s)

@[spec_rule] theorem reciprocal_memory {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (lt : α → α → Bool)
    (hlt : M.compareLt (some .fp32) .real = some lt)
    (stride B : Nat) (s : State α) (hd : DivisionDomain M D lt stride B s) :
    (originalExecution M lt hlt stride B s).val.mem =
      (reciprocalExecution M lt hlt stride B s).val.mem := by
  have values (i : TileIndex [B]) (hi : active s i) :
      (if lt (M.literal (some .fp32) .real 0) (loadedLogit M stride s i) then
        M.binary (some .fp32) .real .div (loadedLogit M stride s i) (loadedRepetition M s)
      else M.binary (some .fp32) .real .mul (loadedLogit M stride s i) (loadedRepetition M s)) =
      (if lt (M.literal (some .fp32) .real 0) (loadedLogit M stride s i) then
        M.binary (some .fp32) .real .mul (loadedLogit M stride s i)
          (M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1)
            (loadedRepetition M s))
      else M.binary (some .fp32) .real .mul (loadedLogit M stride s i) (loadedRepetition M s)) := by
    by_cases positive : lt (M.literal (some .fp32) .real 0) (loadedLogit M stride s i) = Bool.true
    · simp only [positive, ↓reduceIte]
      obtain ⟨hx, hr, hn⟩ := hd i hi positive
      exact FP.ScalarArithmetic.div_mul_rcp R M D hM s _ _ hx hr hn
    · simp [positive]
  simp only [originalExecution, reciprocalExecution,
    VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
    State.setReg_mem]
  simp only [loadedLogit, loadedRepetition, address, active, start, stop] at values
  congr 1
  funext mem i
  by_cases hi : active s i
  · have value := values i hi
    simp only [active, start, stop] at hi
    simp only [hi, ↓reduceIte] at value ⊢
    rw [value]
  · simp only [active, start, stop] at hi
    simp only [hi, ↓reduceIte]

/-- The original source writes only active gathered logits; all other typed
cells are preserved. No arithmetic assumption is used for this frame. -/
theorem original_frame {α : Type} [Inhabited α] (M : Algebra α)
    (lt : α → α → Bool) (hlt : M.compareLt (some .fp32) .real = some lt)
    (stride B : Nat) (s : State α) (r : RegionName) (o : Nat)
    (outside : r ≠ "logits" ∨ ∀ i : TileIndex [B], active s i → o ≠ address stride s i) :
    (originalExecution M lt hlt stride B s).val.mem r o = s.mem r o := by
  simp only [originalExecution]
  apply State.masked_scatter_frame
  rcases outside with hr | ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

/-- Empty metadata windows admit arbitrary values, including a zero or
nonfinite repetition word, because no lane reaches a store. -/
theorem empty_domain {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (lt : α → α → Bool) (stride B : Nat) (s : State α) (empty : stop s ≤ start s) :
    DivisionDomain M D lt stride B s := by
  intro i hi
  unfold active at hi
  omega

/-- Nonpositive and unordered comparisons select the unchanged product arm. -/
theorem unchanged_arm_domain {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (lt : α → α → Bool) (stride B : Nat) (s : State α)
    (unchanged : ∀ i : TileIndex [B], active s i →
      lt (M.literal (some .fp32) .real 0) (loadedLogit M stride s i) = Bool.false) :
    DivisionDomain M D lt stride B s := by
  intro i hi hp
  simp [unchanged i hi] at hp

structure Program where
  stride : Nat
  block : Nat
  reciprocal : Bool

def Program.source (p : Program) : ComputeKernel :=
  if p.reciprocal then reciprocalSource p.stride p.block else originalSource p.stride p.block

/-- Successful execution and complete memory equality on the actual active
positive rewrite domain. Backend comparison availability is explicitly separate
from the transformation rule. The exterior frame is retained on both sides. -/
def Equivalent (R : Spec.Assumptions FP.GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ lt, M.compareLt (some .fp32) .real = some lt →
    ∀ s, DivisionDomain M D lt lhs.stride lhs.block s →
      ∃ a b, FP.Structural.exec (engine M) lhs.source s = some a ∧
        FP.Structural.exec (engine M) rhs.source s = some b ∧ a.mem = b.mem ∧
        (∀ (r : RegionName) o, (r ≠ "logits" ∨ ∀ i : TileIndex [lhs.block],
          active s i → o ≠ address lhs.stride s i) → a.mem r o = s.mem r o ∧ b.mem r o = s.mem r o)

instance : Spec.ProgramSyntax Program where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := (List RegionName × List RegionName) × Nat × Nat
  signature p := ((p.source.inputs, p.source.outputs), p.stride, p.block)
  body p := [⟨[], p.source.surfaceBody, .real, none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun lhs rhs => lhs = rhs

#axiomsClean originalExecution
#axiomsClean reciprocalExecution
#axiomsClean reciprocal_memory
#axiomsClean original_frame
#axiomsClean empty_domain
#axiomsClean unchanged_arm_domain

end
end VeriTile.Bench.Repairs.ApplyPenalty.Rewrite
