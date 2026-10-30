import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ConfiguredPlan
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open CrossEntropy1.Floating
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (TensorArithmetic Observations World)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Storage sign is separate from the expression core. Lazy negative views
therefore have an involution on every source value, including arbitrary NaNs,
raw words and previously computed expressions. -/
structure SignedExpr where
  core : Expr
  negative : Bool := Bool.false

instance : Inhabited SignedExpr := ⟨⟨.raw 0,Bool.false⟩⟩

def SignedExpr.neg (x : SignedExpr) : SignedExpr := {x with negative:=!x.negative}

def SignedExpr.erase (x : SignedExpr) : Expr :=
  if x.negative then .negation x.core else x.core

theorem SignedExpr.neg_neg (x : SignedExpr) : x.neg.neg=x := by
  cases x with | mk core negative => cases negative <;> rfl

def signedFP : FP32 SignedExpr where
  literal := fun x => ⟨.literal x,Bool.false⟩
  fromNat := fun x => ⟨.natural x,Bool.false⟩
  negInf := ⟨.negativeInfinity,Bool.false⟩
  load := fun x => ⟨.barrier x.erase,Bool.false⟩
  add := fun a b => ⟨.binary .add a.erase b.erase,Bool.false⟩
  sub := fun a b => ⟨.binary .sub a.erase b.erase,Bool.false⟩
  mul := fun a b => ⟨.binary .mul a.erase b.erase,Bool.false⟩
  div := fun a b => ⟨.binary .div a.erase b.erase,Bool.false⟩
  exp := fun x => ⟨.unary .exp x.erase,Bool.false⟩
  log := fun x => ⟨.unary .log x.erase,Bool.false⟩
  sum := fun {width} xs => ⟨.sum width (fun i => (xs i).erase),Bool.false⟩
  maximum := fun {width} xs => ⟨.maximum width (fun i => (xs i).erase),Bool.false⟩

def signedHost : TensorArithmetic SignedExpr where
  zero := ⟨.literal 0,Bool.false⟩
  add := fun a b => ⟨.hostAdd (.barrier a.erase) (.barrier b.erase),Bool.false⟩
  square := fun x => ⟨.hostSquare (.barrier x.erase),Bool.false⟩
  scale := fun s x => ⟨.hostScale s (.barrier x.erase),Bool.false⟩
  sum := fun xs => ⟨.hostSum xs.length (fun i => .barrier (xs.get i).erase),Bool.false⟩
  logsumexp := fun xs => ⟨.hostLSE xs.length (fun i => .barrier (xs.get i).erase),Bool.false⟩
  rankSum := fun xs => ⟨.rankSum xs.length (fun i => .barrier (xs.get i).erase),Bool.false⟩

def numericObservations {σ : Type} (integer : World SignedExpr σ → Tensor → Nat → Int) : Observations SignedExpr σ :=
  ⟨SignedExpr.neg,integer⟩

/-- Negation of FP32 storage is the exact sign-bit operation. This is fixed
representation behavior; all other numeric ports retain their old domains. -/
def signNumerics (n : Numerics) : Numerics := {n with neg:=CrossEntropy1.Physical.flipSign}

def signedReference {σ : Type} {selection : Selection σ} (contract : Occurrence σ → Bool)
    {expression : SignedExpr} (plan : SelectedPlan selection expression.core) (n : Numerics) : Word :=
  let core := ConfiguredPlan.reference contract plan (signNumerics n)
  if expression.negative then CrossEntropy1.Physical.flipSign core else core

theorem signedReference_equiv {σ : Type} {selection : Selection σ} (contract : Occurrence σ → Bool)
    {expression : SignedExpr} (plan : SelectedPlan selection expression.core) (native expected : Numerics)
    (ports : Obligations (signNumerics native) (signNumerics expected)) :
    signedReference contract plan native=signedReference contract plan expected := by
  unfold signedReference
  rw [ConfiguredPlan.reference_equiv contract plan _ _ ports]

#axiomsClean SignedExpr.neg_neg
#axiomsClean signedReference_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
