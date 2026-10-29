import bench.repairs.tritonbench_g.isfinite_kernel.Misc.RealEndpoints
import bench.repairs.tritonbench_g.isfinite_kernel.Kernels
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.IsfiniteKernel.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Kernel syntax plus its classification precision. The structural relation
quantifies over every scalar classifier, including classifiers of NaN words. -/
structure Program where
  kernel : ComputeKernel
  precision : Precision

def Equivalent (lhs rhs : Program) : Prop :=
  ∀ (A : Type) [Inhabited A] (classify : A → Bool) (before : State A),
    ∃ left right, FP.Structural.exec (Source.algebra lhs.precision classify) lhs.kernel before = some left ∧
      FP.Structural.exec (Source.algebra rhs.precision classify) rhs.kernel before = some right ∧
      left.mem = right.mem

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := Precision × List RegionName × List RegionName
  signature p := (p.precision,p.kernel.inputs,p.kernel.outputs)
  body p := p.kernel.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs = rhs

def original (precision : Precision) (input output : View) (n B : Nat) : Program :=
  ⟨source precision input output n B,precision⟩

def commuted (precision : Precision) (input output : View) (n B : Nat) : Program :=
  ⟨indexCommuted precision input output n B,precision⟩

/-- Both full kernels succeed and have identical complete memory. Only an
integer multiplication is commuted; no numerical experiment or finite-only
domain is needed, even in masked lanes or when the active input is NaN. -/
theorem isfinite_index_commuted_memory_equiv (precision : Precision)
    (input output : View) (n B : Nat) :
    original precision input output n B ≡[([] : Spec.Assumptions ComputeStmt)]
      commuted precision input output n B := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent) rfl
  · cases precision <;> rfl
  intro A _ classify before
  obtain ⟨left,hl,ml⟩ := Source.executes precision classify input output n B before
  obtain ⟨right,hr,mr⟩ := Source.commuted_executes precision classify input output n B before
  exact ⟨left,right,hl,hr,ml.trans mr.symm⟩

theorem original_source (precision : Precision) (input output : View) (n B : Nat) :
    (original precision input output n B).kernel = source precision input output n B := rfl

/-- Complete native bit-classifier result and source/candidate equality,
including the active outputs and exterior frame in `Kernel.Post`. -/
theorem isfinite_native_result (precision : Precision) (input output : View) (n B : Nat)
    (before : State (Bytes.Word precision.width))
    (injective : ∀ i, i<n → ∀ j, j<n → output.address i=output.address j → i=j) :
    ∃ left right,
      FP.Structural.exec (Source.algebra precision (Bytes.reference precision))
        (source precision input output n B) before=some left ∧
      FP.Structural.exec (Source.algebra precision (Bytes.reference precision))
        (indexCommuted precision input output n B) before=some right ∧
      left.mem=right.mem ∧ Kernel.Post (Bytes.reference precision) input output n B before left := by
  obtain ⟨left,ran,post⟩ := RealEndpoints.isfinite_kernel_real precision input output n B before injective
  obtain ⟨sourceResult,sourceRan,sourceMemory⟩ := Source.executes precision (Bytes.reference precision) input output n B before
  obtain ⟨right,rightRan,rightMemory⟩ := Source.commuted_executes precision (Bytes.reference precision) input output n B before
  have same : left=sourceResult := Option.some.inj (ran.symm.trans sourceRan)
  exact ⟨left,right,ran,rightRan,same ▸ sourceMemory.trans rightMemory.symm,post⟩

#axiomsClean isfinite_native_result
#print_fp_assumptions isfinite_index_commuted_memory_equiv
#axiomsClean isfinite_index_commuted_memory_equiv
#axiomsClean original_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.FPEquiv
