import bench.repairs.tritonbench_g.add_example.Contract
import bench.repairs.tritonbench_g.seeded_dropout.Kernels

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Torch
noncomputable section
abbrev TensorContract (x : TensorMetadata) := AddExample.Contiguous.InputContract x x

/-- A finite constructor contains the exact real value of an actual Python float.
NaNs and infinities have their own constructor, and non-floats are rejected. -/
inductive ProbabilityArgument where
  | other | nonfinite | finite (value : ℝ)
inductive SeedArgument where
  | other | integer (value : Int)

inductive InputError where
  | tensor (reason : AddExample.Contiguous.InputError)
  | probabilityType | probability | seedType | seedRange
  deriving DecidableEq

structure InputContract (x : TensorMetadata) (p : ℝ) (seed : Int) : Prop where
  tensor : TensorContract x
  probability : 0 ≤ p ∧ p < 1
  seedRange : -(2^63 : Int) ≤ seed ∧ seed < 2^64

/-- The installed Python integer branch order, including signed boundary cases. -/
def SeedABI (kind : SeedKind) (seed : SeedCarrier kind) : Prop :=
  let s := seedValue kind seed
  (kind = .i32 ∧ -(2^31 : Int) ≤ s ∧ s < 2^31) ∨
  (kind = .i64 ∧ -(2^63 : Int) ≤ s ∧ s < 2^63 ∧
    (s < -(2^31 : Int) ∨ (2^31 : Int) ≤ s)) ∨
  (kind = .u64 ∧ (2^63 : Int) ≤ s ∧ s < 2^64)

def selectSeed (s : Int) : (kind : SeedKind) × SeedCarrier kind :=
  if -(2^31 : Int) ≤ s ∧ s < 2^31 then ⟨.i32,s⟩
  else if s < 2^63 then ⟨.i64,s⟩ else ⟨.u64,s.toNat⟩

def checkScalars (p : ProbabilityArgument) (seed : SeedArgument) : Except InputError Unit :=
  match p with
  | .other => .error .probabilityType
  | .nonfinite => .error .probability
  | .finite p =>
    if ¬ (0 ≤ p ∧ p < 1) then .error .probability
    else match seed with
    | .other => .error .seedType
    | .integer s => if ¬ (-(2^63 : Int) ≤ s ∧ s < 2^64) then .error .seedRange else .ok ()

def checkInputs (x : ArgumentMetadata) (p : ProbabilityArgument) (seed : SeedArgument) :
    Except InputError Unit :=
  match AddExample.Contiguous.checkInputs x x with
  | .error e => .error (.tensor e)
  | .ok () => checkScalars p seed

def AdmittedArguments (x : ArgumentMetadata) (p : ProbabilityArgument) (seed : SeedArgument) : Prop :=
  match x,p,seed with
  | .tensor x,.finite p,.integer seed => InputContract x p seed
  | _,_,_ => False
end
end VeriTile.Bench.Repairs.SeededDropout
