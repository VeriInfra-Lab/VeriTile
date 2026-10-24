import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SoftwareProfile
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivOps.LocalNumerics
open FPProfile VeriTile.Triton FP.Structural
noncomputable section

abbrev Finite (w : Word) : Prop := w.toRat?.isSome=Bool.true
def real (w : Word) : ℝ := ((w.toRat?.getD 0 : Rat) : ℝ)
def unaryResidual (op : Word→Word) (ideal : ℝ→ℝ) (x : Word) : ℝ := real (op x)-ideal (real x)
def binaryResidual (op : Word→Word→Word) (ideal : ℝ→ℝ→ℝ) (x y : Word) : ℝ :=
  real (op x y)-ideal (real x) (real y)

/-- A local numerical obligation at one actual unary site. This is not a
kernel assumption. The domain and tolerance must be validated separately. -/
def UnaryBound (op : Word→Word) (ideal : ℝ→ℝ) (domain : Word→Prop) (tolerance : Word→ℝ) : Prop :=
  ∀ x,domain x → Finite x → Finite (op x) → |unaryResidual op ideal x|≤tolerance x

def BinaryBound (op : Word→Word→Word) (ideal : ℝ→ℝ→ℝ)
    (domain : Word→Word→Prop) (tolerance : Word→Word→ℝ) : Prop :=
  ∀ x y,domain x y → Finite x → Finite y → Finite (op x y) →
    |binaryResidual op ideal x y|≤tolerance x y

/-- Every actual operand/result is finite, and both relevant log arguments
are positive. Inactive lanes require their own domain if bounds are requested
before remasking; the total execution theorem has no finite restriction. -/
def DefaultDomain (rest : Algebra Word) (eps y target : Word) : Prop :=
  let q := rest.binary (some .fp32) .real .max target eps
  let l := rest.unary (some .fp32) .log q
  let d := FP.sub config l y
  let p := FP.mul config target d
  Finite eps ∧ Finite y ∧ Finite target ∧ Finite q ∧ Finite l ∧ Finite d ∧ Finite p ∧
    0<real q ∧ 0<max (real target) (real eps)

def LogDomain (rest : Algebra Word) (y target : Word) : Prop :=
  let e := rest.unary (some .fp32) .exp target
  let d := FP.sub config target y
  let p := FP.mul config e d
  Finite y ∧ Finite target ∧ Finite e ∧ Finite d ∧ Finite p

/-- Exact error decomposition into the actual max/log/sub/mul site residuals.
No whole-output accuracy or transcendental identity is assumed. -/
theorem default_error (rest : Algebra Word) (eps y target : Word) (_domain : DefaultDomain rest eps y target) :
    let q := rest.binary (some .fp32) .real .max target eps
    let l := rest.unary (some .fp32) .log q
    let d := FP.sub config l y
    let p := FP.mul config target d
    real (Tile.scalar (algebra rest) Bool.false eps y target)-
      RealExecution.loss Bool.false (real eps) (real y) (real target)=
    real target * (Real.log (max (real target) (real eps)+(real q-max (real target) (real eps))) -
      Real.log (max (real target) (real eps)) + (real l-Real.log (real q)) +
      (real d-(real l-real y))) + (real p-real target*real d) := by
  dsimp only
  rw [scalar_default]
  have maximum : max (real target) (real eps)+
      (real (rest.binary (some .fp32) .real .max target eps)-max (real target) (real eps))=
      real (rest.binary (some .fp32) .real .max target eps) := by ring
  rw [maximum]
  simp only [RealExecution.loss,VeriTile.Bench.TritonBenchG.KldivOps.FullMath.loss,Bool.false_eq_true,if_false]
  ring

/-- exp/sub/mul residuals include their cross term; the two source operations
are never silently identified with FMA. -/
theorem log_error (rest : Algebra Word) (eps y target : Word) (_domain : LogDomain rest y target) :
    let e := rest.unary (some .fp32) .exp target
    let d := FP.sub config target y
    let p := FP.mul config e d
    real (Tile.scalar (algebra rest) Bool.true eps y target)-
      RealExecution.loss Bool.true (real eps) (real y) (real target)=
    (real e-Real.exp (real target))*(real target-real y)+
      Real.exp (real target)*(real d-(real target-real y))+
      (real e-Real.exp (real target))*(real d-(real target-real y))+
      (real p-real e*real d) := by
  dsimp only
  rw [scalar_log]
  simp only [RealExecution.loss,VeriTile.Bench.TritonBenchG.KldivOps.FullMath.loss,if_true]
  ring

/-- A Torch sum is one ordinary local primitive. dim0 distinguishes the actual
.sum(dim=0) and .sum() calls; no reassociation of the Triton row graph occurs. -/
def TorchSumResidual (sum : Bool→List Word→Word) (dim0 : Bool) (xs : List Word) : ℝ :=
  real (sum dim0 xs)-(xs.map real).sum

def DivideResidual (divide : Word→Nat→Word) (x : Word) (n : Nat) : ℝ := real (divide x n)-real x/n

#specNonCircular UnaryBound avoiding [Tile.scalar]
#specNonCircular BinaryBound avoiding [Tile.scalar]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.LocalNumerics
