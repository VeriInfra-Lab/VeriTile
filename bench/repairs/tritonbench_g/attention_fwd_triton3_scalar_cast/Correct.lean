import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.LogicalReader
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RealImplementation

/-! Complete ordinary-grid Real specification. RealIO exposes finite six-input
windows, carry aliases, padding, masks and the exact shared source projection.
The original canonical-flat-memory and Tensor results remain in Auxiliary and
TensorBridge, using the same source execution and independent formula.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- Base-two rescaling in the existing Real projection. A bottom center
contributes zero. This is a pure mathematical convention, independent of FP. -/
def rescaling : WithBot ℝ → WithBot ℝ → ℝ
  | .some a, .some b => pow2 (a - b)
  | _, _ => 0

def maximumScore : List (ℝ × ℝ) → WithBot ℝ
  | [] => ⊥
  | x :: xs => (x.1 : WithBot ℝ) ⊔ maximumScore xs

def score (BM BD start : Nat) (p : RealPrologue.Parameters)
    (xs : RealInputs.Inputs BM BD p.layout.kvContext)
    (r : Fin BM) (k : Fin p.layout.kvContext) : ℝ :=
  (∑ d : Fin BD,
    (if p.evenM ≠ 0 ∨ start * BM + r.val < p.layout.queryContext then xs.query r d else 0) *
      xs.key k d) * (p.scale * 1.4426950408889634)

/-- Signed distance; the complement keeps distance ≥ size, independently of
its sign. Disabling the window keeps every logical key. -/
def keepKey (start BM row window complement : Nat) (offset size : Int) (key : Nat) : Bool :=
  if window = 0 then Bool.true
  else if complement = 0 then
    decide (0 ≤ WindowBounds.distance start BM row key offset ∧
      WindowBounds.distance start BM row key offset < size)
  else decide (size ≤ WindowBounds.distance start BM row key offset)

def initial (BM BD : Nat) (p : RealPrologue.Parameters)
    (xs : RealInputs.Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => if p.init ≠ 0 then ⊥ else some (xs.center r)
  denominator := fun r => if p.init ≠ 0 then 1 else xs.denominator r
  numerator := fun i => if p.init ≠ 0 then 0 else xs.output i.1 i.2.1

/-- A closed maximum and two sums over the finite kept keys. No-loop returns
its seed; executing an all-masked block instead uses the empty-pair formula. -/
def rowSummary (maximum : List (ℝ × ℝ) → WithBot ℝ) (BM BN BD start : Nat) (p : RealPrologue.Parameters)
    (xs : RealInputs.Inputs BM BD p.layout.kvContext) (r : Fin BM)
    (initialValue : ℝ) (value : Fin p.layout.kvContext → ℝ) : RealFormula.RowState :=
  let seed := RealLoopFormula.rowSeed (initial BM BD p xs) r initialValue
  if (RealKeys.lower start BM BN p.window p.complement p.offset p.size : Int) <
      RealWindow.upper start BM BN p.layout.kvContext p.window p.complement p.offset then
    let pairs := ((List.finRange p.layout.kvContext).filter
      (fun k => keepKey start BM r.val p.window p.complement p.offset p.size k.val)).map
      (fun k => (score BM BD start p xs r k, value k))
    let m := seed.center ⊔ maximum pairs
    { center := m,
      denominator := seed.denominator * rescaling seed.center m +
        (pairs.map (fun pair => rescaling (pair.1 : WithBot ℝ) m * 1)).sum,
      numerator := seed.numerator * rescaling seed.center m +
        (pairs.map (fun pair => rescaling (pair.1 : WithBot ℝ) m * pair.2)).sum }
  else seed

def carry (maximum : List (ℝ × ℝ) → WithBot ℝ) (BM BN BD start : Nat) (p : RealPrologue.Parameters)
    (xs : RealInputs.Inputs BM BD p.layout.kvContext) : RealLoopFormula.Carry BM BD where
  center := fun r => (rowSummary maximum BM BN BD start p xs r 0 (fun _ => 0)).center
  denominator := fun r => (rowSummary maximum BM BN BD start p xs r 0 (fun _ => 0)).denominator
  numerator := fun i => (rowSummary maximum BM BN BD start p xs i.1
    ((initial BM BD p xs).numerator i) (fun k => xs.value k i.2.1)).numerator

/-- Continuation writes M/L/O. END writes M+log₂(L) and O/L while the IO masks
out the L store. Both mathematical division and log retain their total Real meaning. -/
def expected (maximum : List (ℝ × ℝ) → WithBot ℝ) (BM BN BD finish start : Nat) (p : RealPrologue.Parameters)
    (xs : RealInputs.Inputs BM BD p.layout.kvContext) : RealIO.Outputs BM BD :=
  let c := carry maximum BM BN BD start p xs
  { center := fun r => (c.center r).elim 0 (fun m =>
      if decide (finish ≠ 0) then m + Real.log (c.denominator r) / Real.log 2 else m),
    denominator := c.denominator,
    output := fun i => if finish ≠ 0 then c.numerator i / c.denominator i.1 else c.numerator i }

def formula (maximum : List (ℝ × ℝ) → WithBot ℝ) (c : Call) (xs : LogicalIO.Inputs c) : LogicalIO.Outputs c where
  center := fun program row =>
    (expected maximum c.BM c.BN c.D c.finish (c.tile program) c.params (xs.programs program)).center row
  denominator := fun program row =>
    (expected maximum c.BM c.BN c.D c.finish (c.tile program) c.params (xs.programs program)).denominator row
  output := fun program row col =>
    (expected maximum c.BM c.BN c.D c.finish (c.tile program) c.params (xs.programs program)).output (row,col,())

specification correct (c : Call) (valid : Valid c) :
    Spec.Real (LogicalIO.io c ⊨ formula maximumScore c) :=
  LogicalReader.correct_with_maximum c valid maximumScore rfl (fun _ _ => rfl)
    (fun maximum => formula maximum c) (by intros; rfl) (by intros; rfl) (by intros; rfl)

#axiomsClean correct
#specNonCircular formula avoiding
  [Kernels.source,exec,execR,Kernel.GridLaunchedOrdinary,AlgorithmExecution]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
