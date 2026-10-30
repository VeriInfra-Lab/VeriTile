import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeHost
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.WholeFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite FP.Guarded Math Program
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def typed32 (c : Whole.Host) (r : Regions) (swapped hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) :=
  kernel c.layout r .fp32 swapped .fp32 .fp32 hasBias useSoftplus lower upper

def site32 (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) : Site :=
  .mulBroadcastFP32 (Source.prefixBefore (typed32 c r Bool.false hasBias useSoftplus lower upper).surfaceBody "dA")
    [c.BH,c.BC] [c.BH,1] (.ref .real [c.BH,c.BC] "dt")
    (.expandDim ⟨1,by simp⟩ (.ref .real [c.BH] "A"))

theorem program_eq {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : FP.Structural.Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (s : State α) (guard : (site32 c r hasBias useSoftplus lower upper).Holds M D s) :
    FP.Structural.exec (engine M) (typed32 c r Bool.false hasBias useSoftplus lower upper) s =
    FP.Structural.exec (engine M) (typed32 c r Bool.true hasBias useSoftplus lower upper) s := by
  let before := Source.prefixBefore (typed32 c r Bool.false hasBias useSoftplus lower upper).surfaceBody "dA"
  let after := (typed32 c r Bool.false hasBias useSoftplus lower upper).surfaceBody.drop (before.length+1)
  have h := contextual_mul_broadcast_fp32_commute R M D hM s before after "dA"
    (.consSame (.consR .nil)) (.ref .real [c.BH,c.BC] "dt")
    (.expandDim ⟨1,by simp⟩ (.ref .real [c.BH] "A")) guard
  cases upper <;> simpa [Contextual,before,after,typed32,VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Program.kernel,source,finiteSource,infiniteSource,
    Source.prefixBefore,ComputeKernel.surfaceBody,reverseBroadcast,FP.Structural.exec] using h

def tileState {α : Type} (l : Launch) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then l.batch else if ax = 1 then l.chunk
    else if ax = 2 then l.head else 0 }

def runTiles {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (k : ComputeKernel) :
    List Launch → State α → Option (State α)
  | [],s => some s
  | l::rest,s => (FP.Structural.exec (engine M) k (tileState l s)).bind (runTiles M k rest)

/-- Only the actual multiplication operands reached after the common prefix
must be finite. No postcondition, execution-success or whole-result premise. -/
def GridDomain {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (D : Domain α)
    (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    List Launch → State α → Prop
  | [],_ => True
  | l::rest,s => (site32 c r hasBias useSoftplus lower upper).Holds M D (tileState l s) ∧
      ∀ t, FP.Structural.exec (engine M) (typed32 c r Bool.false hasBias useSoftplus lower upper)
        (tileState l s) = some t → GridDomain M D c r hasBias useSoftplus lower upper rest t

theorem grid_eq {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : FP.Structural.Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (ls : List Launch) (s : State α)
    (guard : GridDomain M D c r hasBias useSoftplus lower upper ls s) :
    runTiles M (typed32 c r Bool.false hasBias useSoftplus lower upper) ls s =
    runTiles M (typed32 c r Bool.true hasBias useSoftplus lower upper) ls s := by
  induction ls generalizing s with
  | nil => rfl
  | cons l rest ih =>
    have eq := program_eq R M D hM c r hasBias useSoftplus lower upper (tileState l s) guard.1
    simp only [runTiles,← eq]
    cases he : FP.Structural.exec (engine M) (typed32 c r Bool.false hasBias useSoftplus lower upper)
      (tileState l s) with
    | none => rfl
    | some t => exact ih t (guard.2 t he)

structure GridProgram where
  host : Whole.Host
  regions : Regions
  hasBias : Bool
  useSoftplus : Bool
  lower : ℝ
  upper : UpperBound
  schedule : List Launch
  swapped : Bool

def GridProgram.kernel (p : GridProgram) :=
  typed32 p.host p.regions p.swapped p.hasBias p.useSoftplus p.lower p.upper

def Equivalent (rules : Spec.Assumptions FP.GuardedFragment) (p q : GridProgram) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : FP.Structural.Algebra α) (D : Domain α), Models rules M D →
    ∀ s, GridDomain M D p.host p.regions p.hasBias p.useSoftplus p.lower p.upper p.schedule s →
      runTiles M p.kernel p.schedule s = runTiles M q.kernel q.schedule s

instance : Spec.ProgramSyntax GridProgram where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := Whole.Host × Regions × Bool × Bool × ℝ × UpperBound × List Launch
  signature p := (p.host,p.regions,p.hasBias,p.useSoftplus,p.lower,p.upper,p.schedule)
  body p := [⟨[],p.kernel.surfaceBody,.real,none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun p q => p = q

def originalGrid (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (ls : List Launch) : GridProgram :=
  ⟨c,r,hasBias,useSoftplus,lower,upper,ls,Bool.false⟩

def candidateGrid (c : Whole.Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (ls : List Launch) : GridProgram :=
  ⟨c,r,hasBias,useSoftplus,lower,upper,ls,Bool.true⟩

theorem grid_equivalent (R : FP.ScalarArithmetic.Rules) (c : Whole.Host) (r : Regions)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (ls : List Launch) :
    Spec.FloatingPoint R.assumptions (originalGrid c r hasBias useSoftplus lower upper ls)
      (candidateGrid c r hasBias useSoftplus lower upper ls) := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := originalGrid c r hasBias useSoftplus lower upper ls)
    (rhs := candidateGrid c r hasBias useSoftplus lower upper ls)
    (structural := fun _ _ => False) rfl (by rfl)
  intro α _ M D model s guard
  exact grid_eq R M D model c r hasBias useSoftplus lower upper ls s guard

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.WholeFP
