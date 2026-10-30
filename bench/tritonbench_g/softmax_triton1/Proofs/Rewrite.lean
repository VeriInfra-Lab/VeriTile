import bench.tritonbench_g.softmax_triton1.Proofs.FP
import bench.tritonbench_g.vector_addition.Sources.Mathematical_Value
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Triton.Float.ScalarArithmeticLaws

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
open FP.ProfiledRewrite (engine)
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- A division site observed only through its output mask. The shared scalar
is still used by every active lane; padded numerator values are unrestricted. -/
structure DivisionSite where
  before : List ComputeStmt
  lanes : Nat
  active : Op .bool [lanes]
  numerator : Op .real [lanes]
  denominator : Op .real []

def DivisionSite.Holds {α : Type} [Inhabited α] (site : DivisionSite)
    (M : Algebra α) (D : Domain α) (s : State α) : Prop :=
  ∀ t, run (engine M) site.before s = some t →
    ∀ active a b, evalOp (engine M) none site.active t = some active →
      evalOp (engine M) (some .fp32) site.numerator t = some a →
      evalOp (engine M) (some .fp32) site.denominator t = some b →
      (∀ i, active i = Bool.true → D .finite (a i)) ∧
        D .finite (b PUnit.unit) ∧ D .nonzero (b PUnit.unit)

/-- Whole-memory observation permits irrelevant final registers to differ. -/
structure Program where
  kernel : ComputeKernel
  domain : DivisionSite

def Equivalent (R : Spec.Assumptions FP.GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, lhs.domain.Holds M D s →
      ∃ a b, FP.Structural.exec (engine M) lhs.kernel s = some a ∧
        FP.Structural.exec (engine M) rhs.kernel s = some b ∧ a.mem = b.mem

instance : Spec.ProgramSyntax Program where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := (List RegionName × List RegionName) × DivisionSite
  signature p := ((p.kernel.inputs, p.kernel.outputs), p.domain)
  body p := [⟨[], p.kernel.surfaceBody, .real, none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun lhs rhs => lhs = rhs

/-- The original source prefix includes both complete padded reductions. -/
def sourcePrefix (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) : List ComputeStmt :=
  Source.prefixBefore (softmax_kernel out inp ib ob ir ic orow oc n B).surfaceBody
    "softmax_output"

/-- The actual final store mask selects the numerator lanes requiring the
numerical law. This transformation domain is not a Python input guard. -/
def divisionSite (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) : DivisionSite where
  before := sourcePrefix out inp ib ob ir ic orow oc n B
  lanes := B
  active := .lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n)
  numerator := .ref .real [B] "numerator"
  denominator := .ref .real [] "denominator"

/-- Successful execution of the actual original source, without numerical laws. -/
def originalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (s : State α) :
    {t : State α // FP.Structural.exec (engine M)
      (softmax_kernel out inp ib ob ir ic orow oc n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [FP.Structural.exec,softmax_kernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,State.setReg,numeric,natLt,bop,
    TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType]
  rfl

/-- The reciprocal source also succeeds independently of the numerical law. -/
def reciprocalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (s : State α) :
    {t : State α // FP.Structural.exec (engine M)
      (reciprocal_kernel out inp ib ob ir ic orow oc n B) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [FP.Structural.exec,reciprocal_kernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,store,State.setReg,numeric,natLt,bop,
    TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType]
  rfl

/-- Values reached by the common prefix, retaining the complete padded max
and sum. These definitions are checked against actual execution below. -/
def numeratorValue {α : Type} [Inhabited α] (M : Algebra α)
    (inp : RegionName) (ib ir ic n B : Nat) (s : State α) : Values α .real [B] :=
  let row : Values α .real [B] := fun j => M.fp32Load
    (if j.1.val < n then (s.mem inp (ib+s.pids 0*ir+j.1.val*ic)).read .real else M.negInf)
  let maximum := M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false row PUnit.unit
  fun j => M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (row j) maximum)

def denominatorValue {α : Type} [Inhabited α] (M : Algebra α)
    (inp : RegionName) (ib ir ic n B : Nat) (s : State α) : α :=
  M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false (numeratorValue M inp ib ir ic n B s) PUnit.unit

/-- Exactly active numerator finiteness and shared denominator validity;
there is no premise on the final division's inactive numerator lanes. -/
theorem site_values_iff {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (s : State α) :
    (divisionSite out inp ib ob ir ic orow oc n B).Holds M D s ↔
      (∀ i : TileIndex [B], i.1.val < n → D .finite (numeratorValue M inp ib ir ic n B s i)) ∧
        D .finite (denominatorValue M inp ib ir ic n B s) ∧
        D .nonzero (denominatorValue M inp ib ir ic n B s) := by
  simp [divisionSite, DivisionSite.Holds, sourcePrefix, Source.prefixBefore,
    softmax_kernel, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, numeric, natLt, bop, Region.cast, State.setReg,
    TileShape.axisDim, hB, ComputeDType.eraseDType,
    engine, Algebra.withDefaultPrecision, resolvePrecision, numeratorValue, denominatorValue, Function.comp_def]

/-- Both complete sources agree on every typed memory cell. Only active final
divisions use DIV-MUL-RCP; no equality is asserted for dead padded registers. -/
@[spec_rule] theorem reciprocal_memory {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (model : Models R.assumptions M D)
    (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) (s : State α)
    (domain : (divisionSite out inp ib ob ir ic orow oc n B).Holds M D s) :
    (originalExecution M out inp ib ob ir ic orow oc n B hB s).val.mem =
      (reciprocalExecution M out inp ib ob ir ic orow oc n B hB s).val.mem := by
  have finite := (site_values_iff M D out inp ib ob ir ic orow oc n B hB s).mp domain
  have values (i : TileIndex [B]) (hi : i.1.val < n) :
      M.binary (some .fp32) .real .div (numeratorValue M inp ib ir ic n B s i)
        (denominatorValue M inp ib ir ic n B s) =
      M.binary (some .fp32) .real .mul (numeratorValue M inp ib ir ic n B s i)
        (M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1)
          (denominatorValue M inp ib ir ic n B s)) :=
    FP.ScalarArithmetic.div_mul_rcp R M D model s _ _ (finite.1 i hi) finite.2.1 finite.2.2
  simp only [originalExecution, reciprocalExecution, bop, Broadcast.leftIndex,
    Broadcast.rightIndex, decide_eq_true_eq,
    VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory,
    State.setReg_mem]
  simp only [engine, Algebra.withDefaultPrecision, resolvePrecision,
    Option.getD_some, Option.getD_none]
  congr 1
  funext mem i
  by_cases hi : i.1.val < n
  · simp only [hi, ↓reduceIte]
    have eq := values i hi
    simp only [numeratorValue, denominatorValue, hi, ↓reduceIte] at eq
    exact congrArg (fun value => fun r o =>
      if r = out ∧ o = ob+s.pids 0*orow+i.1.val*oc
      then Cell.mk .real value else mem r o) eq
  · simp only [hi, ↓reduceIte]

/-- Public floating judgment assembled from the existing complete executions
and memory equality. Only the actual masked division-site domain is assumed. -/
theorem equivalence (R : FP.ScalarArithmetic.Rules) (out inp : RegionName)
    (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) :
    ({ kernel := softmax_kernel out inp ib ob ir ic orow oc n B,
       domain := divisionSite out inp ib ob ir ic orow oc n B } : Program) ≡[R]
    ({ kernel := reciprocal_kernel out inp ib ob ir ic orow oc n B,
       domain := divisionSite out inp ib ob ir ic orow oc n B } : Program) := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D model s domain
  let a := originalExecution M out inp ib ob ir ic orow oc n B hB s
  let b := reciprocalExecution M out inp ib ob ir ic orow oc n B hB s
  exact ⟨a.val,b.val,a.property,b.property,
    reciprocal_memory R M D model out inp ib ob ir ic orow oc n B hB s domain⟩


#axiomsClean equivalence
#axiomsClean originalExecution
#axiomsClean reciprocalExecution
#axiomsClean site_values_iff
#axiomsClean reciprocal_memory
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Rewrite
