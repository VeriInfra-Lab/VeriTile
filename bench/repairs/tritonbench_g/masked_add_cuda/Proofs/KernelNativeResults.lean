import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Variant
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Contract
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Binding

/-! Kernel-only equivalence of the real addition-order variant. In the selected
native lowering both sources consume the product in the same single FMA, in
alpha/data/grad order. No split-rounding ADD-COMMUTE rule or primitive IEEE
conformance experiment is assumed. All FP32 words are admitted. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.KernelNativeResults
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
open scoped VeriTile.Spec
noncomputable section
abbrev Word := BitVec 32

structure Program where
  storage : MaskStorage
  candidate : Bool
  n : Nat
  alpha : ℝ

def Program.source (p : Program) : ComputeKernel :=
  if p.candidate then commutedKernel p.storage "grad" "data" "mask" p.n p.alpha
  else kernel p.storage "grad" "data" "mask" p.n p.alpha

/-- Both complete sources must succeed and agree on every typed memory cell.
The leaf condition binds loaded FP32 words; it excludes no bit patterns. -/
def Equivalent (lhs rhs : Program) : Prop :=
  ∀ (rest : Algebra Graph.Expr) (alphaWord : Word) (before : State Graph.Expr) (grad : Nat → Word),
    (∀ i : TileIndex [blockSize], active (parameters "grad" "data" "mask" lhs.n) before i →
      (before.mem "grad" (offset (parameters "grad" "data" "mask" lhs.n) before i)).read .real =
        .leaf (grad (offset (parameters "grad" "data" "mask" lhs.n) before i))) →
    ∃ originalFinal commutedFinal,
      FP.Structural.exec (Graph.nativeAlgebra rest alphaWord) lhs.source before = some originalFinal ∧
      FP.Structural.exec (Graph.nativeAlgebra rest alphaWord) rhs.source before = some commutedFinal ∧
      originalFinal.mem = commutedFinal.mem

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := MaskStorage × Nat × ℝ
  signature p := (p.storage, p.n, p.alpha)
  body p := p.source.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs = rhs

def original (storage : MaskStorage) (n : Nat) (alpha : ℝ) : Program := ⟨storage, Bool.false, n, alpha⟩
def commuted (storage : MaskStorage) (n : Nat) (alpha : ℝ) : Program := ⟨storage, Bool.true, n, alpha⟩
def assumptions : Spec.Assumptions ComputeStmt := []

theorem masked_add_commuted_equiv (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    original storage n alpha ≡[assumptions] commuted storage n alpha := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := original storage n alpha) (rhs := commuted storage n alpha) rfl rfl
  intro rest alphaWord before grad hg
  exact Variant.equivalent rest alphaWord storage (parameters "grad" "data" "mask" n)
    alpha before grad hg

#print_fp_assumptions masked_add_commuted_equiv
#axiomsClean masked_add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.KernelNativeResults
