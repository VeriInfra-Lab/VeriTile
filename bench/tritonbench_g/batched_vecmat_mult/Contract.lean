import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_HostPrepare

/-! Exact classification of the eight Python scalar arguments. `.other`
includes bool, non-builtin integer objects, floats, and arbitrary objects.
The wrapper checks all eight before making its first allocation. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Contract
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs

inductive Scalar where
  | integer (value : Int)
  | other
  deriving DecidableEq, Repr

def Scalar.valid : Scalar → Prop
  | .integer n => 0 ≤ n
  | .other => False

instance (s : Scalar) : Decidable s.valid := by
  cases s <;> unfold Scalar.valid <;> infer_instance

def Scalar.toNat : Scalar → Nat
  | .integer n => n.toNat
  | .other => 0

structure Arguments where
  (M N K BM BN BK numWarps numStages : Scalar)

def InputContract (a : Arguments) : Prop :=
  a.M.valid ∧ a.N.valid ∧ a.K.valid ∧ a.BM.valid ∧ a.BN.valid ∧
    a.BK.valid ∧ a.numWarps.valid ∧ a.numStages.valid

instance (a : Arguments) : Decidable (InputContract a) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def toArguments (a : Arguments) : Host.Arguments :=
  ⟨a.M.toNat, a.N.toNat, a.K.toNat, a.BM.toNat, a.BN.toNat, a.BK.toNat,
    a.numWarps.toNat, a.numStages.toNat⟩

def ofArguments (a : Host.Arguments) : Arguments :=
  ⟨.integer a.M, .integer a.N, .integer a.K, .integer a.BM, .integer a.BN,
    .integer a.BK, .integer a.numWarps, .integer a.numStages⟩

def checkInputs (a : Arguments) : Option Host.Arguments :=
  if InputContract a then some (toArguments a) else none

theorem python_checks_iff_preconditions (a : Arguments) :
    (checkInputs a).isSome = Bool.true ↔ InputContract a := by
  simp [checkInputs]

theorem toNat_exact (s : Scalar) (h : s.valid) :
    Scalar.integer s.toNat = s := by
  cases s with
  | other => exact False.elim h
  | integer n => simp [Scalar.toNat,Int.toNat_of_nonneg h]

theorem admitted_roundtrip (a : Arguments) (h : InputContract a) :
    ofArguments (toArguments a) = a := by
  rcases h with ⟨hm,hn,hk,hbm,hbn,hbk,hw,hs⟩
  cases a
  simp only [toArguments,ofArguments,toNat_exact _ hm,toNat_exact _ hn,
    toNat_exact _ hk,toNat_exact _ hbm,toNat_exact _ hbn,toNat_exact _ hbk,
    toNat_exact _ hw,toNat_exact _ hs]

theorem natural_arguments_admitted (a : Host.Arguments) :
    InputContract (ofArguments a) := by
  simp [InputContract,ofArguments,Scalar.valid]

@[simp] theorem natural_arguments_roundtrip (a : Host.Arguments) :
    toArguments (ofArguments a) = a := by
  cases a
  simp [toArguments,ofArguments,Scalar.toNat]

/-- The complete accepted scalar domain is exactly the natural host metadata;
no positive-block or hardware-size check has been hidden in classification. -/
theorem admitted_iff_natural_arguments (a : Arguments) :
    InputContract a ↔ ∃ n : Host.Arguments, a = ofArguments n := by
  constructor
  · intro h
    exact ⟨toArguments a,(admitted_roundtrip a h).symm⟩
  · rintro ⟨n,rfl⟩
    exact natural_arguments_admitted n

#axiomsClean python_checks_iff_preconditions
#axiomsClean admitted_iff_natural_arguments
end VeriTile.Bench.Repairs.BatchedVecmatMult.Contract
